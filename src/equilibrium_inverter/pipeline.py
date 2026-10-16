"""End-to-end construction and target-free MAP prediction API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .config import EquilibriumInverterConfig
from .geometry.compiler import MachineOperator, compile_machine_operator
from .inference.map import MAPInference
from .initialization.proposal import InitialStateProposal
from .inputs.shot import PreparedShot, prepare_shot, to_torch_shot
from .physics.forward import EquilibriumForwardModel
from .probability.posterior import PosteriorEnergy
from .state.latent import StateLayout


_FRAMEWORK_TAG = "equilibrium_inverter"
_LEGACY_FRAMEWORK_TAG = "bayesian_equilibrium"


@dataclass(frozen=True)
class EquilibriumPrediction:
    psi_Wb_per_rad: np.ndarray
    q95: np.ndarray
    beta_n: np.ndarray
    topology_probabilities: np.ndarray
    latent_vector: np.ndarray
    posterior_terms: dict[str, float]
    geometry_manifest: dict[str, object]
    loss_history: tuple[float, ...]


class EquilibriumInverter:
    """Reusable framework whose weights are independent of tokamak identity."""

    def __init__(
        self,
        config: EquilibriumInverterConfig | None = None,
        *,
        device: torch.device | str = "cpu",
        proposal: InitialStateProposal | None = None,
    ) -> None:
        self.config = config or EquilibriumInverterConfig()
        self.device = torch.device(device)
        self.layout = StateLayout.from_solver_config(self.config.solver)
        self.proposal = proposal or InitialStateProposal(self.layout)
        self.proposal.to(self.device)

    def compile(
        self,
        row: Mapping[str, object],
        *,
        frame_indices: np.ndarray | None = None,
    ) -> tuple[PreparedShot, MachineOperator]:
        prepared = prepare_shot(row, self.config, frame_indices=frame_indices)
        operator = compile_machine_operator(prepared, self.config)
        return prepared, operator

    def predict(
        self,
        row: Mapping[str, object],
        *,
        frame_indices: np.ndarray | None = None,
        optimize: bool = True,
    ) -> EquilibriumPrediction:
        prepared, operator = self.compile(row, frame_indices=frame_indices)
        shot = to_torch_shot(
            prepared,
            length_scale_m=operator.length_scale_m,
            device=self.device,
        )
        torch_operator = operator.to_torch(device=self.device)
        forward = EquilibriumForwardModel(torch_operator, self.layout, self.config.solver).to(
            self.device
        )
        proposal = self.proposal(shot, length_scale_m=operator.length_scale_m)
        posterior = PosteriorEnergy(self.layout, self.config.posterior).to(self.device)
        if optimize:
            result = MAPInference(forward, posterior, self.config.inference).run(shot, proposal)
            latent = result.latent_vector
            prediction = result.forward
            posterior_terms = result.posterior_terms
            history = result.loss_history
        else:
            latent = proposal.mean
            prediction = forward(shot, latent)
            posterior_terms = posterior(
                shot,
                latent,
                prediction,
                proposal_mean=proposal.mean.detach(),
                proposal_log_scale=proposal.log_scale.detach(),
            ).detached()
            history = ()
        return EquilibriumPrediction(
            psi_Wb_per_rad=prediction.psi_Wb_per_rad.detach().cpu().numpy(),
            q95=prediction.q95.detach().cpu().numpy(),
            beta_n=prediction.beta_n.detach().cpu().numpy(),
            topology_probabilities=prediction.topology_probabilities.detach().cpu().numpy(),
            latent_vector=latent.detach().cpu().numpy(),
            posterior_terms=posterior_terms,
            geometry_manifest=operator.manifest(),
            loss_history=history,
        )

    def save_proposal(self, path: str) -> None:
        torch.save(
            {
                "framework": _FRAMEWORK_TAG,
                "config": self.config.to_dict(),
                "proposal_state_dict": self.proposal.state_dict(),
            },
            path,
        )

    @classmethod
    def load_proposal(
        cls,
        path: str,
        *,
        device: torch.device | str = "cpu",
    ) -> "EquilibriumInverter":
        """Restore shared DIII-D-trained proposal weights for any geometry."""

        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("framework") not in {_FRAMEWORK_TAG, _LEGACY_FRAMEWORK_TAG}:
            raise ValueError("checkpoint is not an Equilibrium Inverter artifact")
        config = EquilibriumInverterConfig.from_dict(payload["config"])
        inverter = cls(config, device=device)
        inverter.proposal.load_state_dict(payload["proposal_state_dict"], strict=True)
        return inverter


# Compatibility for imports made before the model rename.
BayesianEquilibriumFramework = EquilibriumInverter
