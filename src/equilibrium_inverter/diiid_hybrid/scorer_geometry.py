"""Official-contour forward geometry with an explicitly approximate backward.

The scorer remains unmodified. Each forward re-extracts the prediction's own
last closed contour. The backward freezes topology, polygon membership and
contour parameterization, and uses implicit normal motion of that contour.
It is NOT an exact derivative through the discontinuous contour search.
Targets belong to the loss only; nothing here is an inference feature.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

SCORER = Path(__file__).resolve().parents[3] / "fusion_scoring"
if str(SCORER) not in sys.path:
    sys.path.insert(0, str(SCORER))
from contour import grid_to_physical, resample_closed
from derive import SCALAR_NAMES, PLAUSIBLE, derive_frame, _points_in_polygon, _bilinear
from lcfs import _best_contour_at
from o_point import find_o_point


def smooth_backward(field, sigma):
    """Keep the exact field forward, but spread its scorer-loss gradient locally.

    The scorer still sees precisely ``field``. This is an explicitly approximate
    backward through a separable Gaussian with reflecting boundaries.
    """
    if sigma <= 0:
        return field
    if field.ndim != 3:
        raise ValueError("expected batch x Z x R fields")
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    x = field[:, None]
    horizontal = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"),
                          kernel.reshape(1, 1, 1, -1))
    blurred = F.conv2d(F.pad(horizontal, (0, 0, radius, radius), mode="reflect"),
                       kernel.reshape(1, 1, -1, 1))[:, 0]
    return field.detach() + (blurred - blurred.detach())


def capture_frame(field, r, z, envelope):
    """Replay the vendored 22-step search, retaining its active level/branch.

    Canonical FEC1 fields have their axis at a maximum (axis_sign=+1).
    Tests compare this contour and the derived values to the unmodified scorer.
    """
    field = np.asarray(field, dtype=np.float64)
    r, z = np.asarray(r, float), np.asarray(z, float)
    if field.shape != (len(z), len(r)) or not np.isfinite(field).all():
        raise ValueError("expected a finite canonical DIII-D field on the public grid")
    mask = np.asarray(envelope, bool)
    mask_f = mask.astype(float)
    axis_r, axis_z, iz, ir, _ = find_o_point(field, r, z, mask)
    border = np.concatenate([field[0], field[-1], field[:,0], field[:,-1]])
    lo, hi = float(border.min()), float(field[iz,ir])-1e-9
    raw = _best_contour_at(field, hi, iz, ir, mask_f) if hi > lo else None
    if raw is not None:
        for _ in range(22):
            mid = .5*(lo+hi)
            candidate = _best_contour_at(field, mid, iz, ir, mask_f)
            if candidate is not None:
                hi, raw = mid, candidate
            else:
                lo = mid
    points = None if raw is None else resample_closed(grid_to_physical(raw, r, z), 512)
    values = derive_frame(field, r, z, "DIII-D", mask, mask_f, contour=points, axis_sign=1.)
    scalars = np.asarray([values[name] for name in SCALAR_NAMES])
    if points is None:
        return dict(points=None, scalars=scalars, axis_cell=(iz,ir),
                    support=None, support_kind="failed", level=None, inside=None)
    rr, zz = np.meshgrid(r,z)
    inside = _points_in_polygon(rr.ravel(),zz.ravel(),points).reshape(field.shape)
    # At a grid-connectivity saddle the active level approaches a grid value.
    # At a wall contact use the most envelope-constrained contour point instead.
    raw_points = grid_to_physical(raw,r,z)
    mask_values = _bilinear(mask_f,r,z,raw_points[:,0],raw_points[:,1])
    difference = np.where(mask, np.abs(field-hi), np.inf)
    critical = np.unravel_index(difference.argmin(),field.shape)
    span = max(abs(float(field[iz,ir])-hi),1e-10)
    if mask_values.min() > .5005 and difference[critical] < max(2e-6*span,2e-9):
        support = np.array([r[critical[1]],z[critical[0]]])
        kind = "grid_saddle"
    elif mask_values.min() < .501:
        support = raw_points[int(mask_values.argmin())]
        kind = "envelope_contact"
    else:
        # This branch is explicitly recorded and must be checked by perturbation
        # experiments; it is not silently treated as an exact saddle derivative.
        dz, dr = np.gradient(field,z,r)
        strength = (_bilinear(dr,r,z,raw_points[:,0],raw_points[:,1])**2
                    +_bilinear(dz,r,z,raw_points[:,0],raw_points[:,1])**2)
        support = raw_points[int(strength.argmin())]
        kind = "minimum_gradient_approximation"
    return dict(points=points, scalars=scalars, axis_cell=(iz,ir), support=support,
                support_kind=kind, level=hi, inside=inside,
                level_grid_gap=float(difference[critical]), mask_min=float(mask_values.min()))


def sample(field, points, r, z, *, derivative=False):
    """Scorer-compatible bilinear interpolation with optional spatial gradient."""
    # Torch searchsorted supports the exact same nonuniform-grid mapping too.
    pr = points[...,0].clamp(float(r[0]),float(r[-1]))
    pz = points[...,1].clamp(float(z[0]),float(z[-1]))
    ir = torch.searchsorted(r,pr.contiguous(),right=True).sub(1).clamp(0,len(r)-2)
    iz = torch.searchsorted(z,pz.contiguous(),right=True).sub(1).clamp(0,len(z)-2)
    tr, tz = (pr-r[ir])/(r[ir+1]-r[ir]),(pz-z[iz])/(z[iz+1]-z[iz])
    a,b,c,d = field[iz,ir],field[iz,ir+1],field[iz+1,ir],field[iz+1,ir+1]
    value = a*(1-tr)*(1-tz)+b*tr*(1-tz)+c*(1-tr)*tz+d*tr*tz
    if not derivative:
        return value
    dr = ((b-a)*(1-tz)+(d-c)*tz)/(r[ir+1]-r[ir])
    dz = ((c-a)*(1-tr)+(d-b)*tr)/(z[iz+1]-z[iz])
    return value,torch.stack((dr,dz),-1)


def axis_position(field, cell, r, z):
    iz,ir = cell
    def offset(a,b,c):
        den = a-2*b+c
        safe = torch.where(den == 0,torch.ones_like(den),den)
        return torch.where(den == 0,torch.zeros_like(den),(.5*(a-c)/safe).clamp(-1,1))
    dr = offset(field[iz,ir-1],field[iz,ir],field[iz,ir+1]) if 0 < ir < field.shape[1]-1 else field.sum()*0
    dz = offset(field[iz-1,ir],field[iz,ir],field[iz+1,ir]) if 0 < iz < field.shape[0]-1 else field.sum()*0
    return torch.stack((r[ir]+dr*(r[1]-r[0]),z[iz]+dz*(z[1]-z[0])))


def contour_scalars(field, points, inside, r, z, axis):
    cr,cz = points[:,0],points[:,1]
    rmax,rmin = cr.max(),cr.min()
    width = (rmax-rmin).clamp_min(1e-12)
    rgeo = .5*(rmax+rmin)
    kappa = (cz.max()-cz.min())/width
    tritop = 2*(rgeo-cr[cz.argmax()])/width
    tribot = 2*(rgeo-cr[cz.argmin()])/width
    rn,zn = cr.roll(-1),cz.roll(-1)
    volume = (torch.pi/3*((cr+rn)*(cr*zn-rn*cz)).sum()).abs()
    dz,dr = torch.gradient(field,spacing=(z,r))
    bp = torch.linalg.vector_norm(torch.stack((dr,dz),-1),dim=-1)/r[None]
    weights = inside.to(field.dtype)*r[None]
    mean_bp2 = (bp.square()*weights).sum()/weights.sum().clamp_min(1e-12)
    boundary_bp = sample(bp,points,r,z)
    segment = torch.linalg.vector_norm(points.roll(-1,0)-points,dim=-1)
    average_bp = (.5*(boundary_bp+boundary_bp.roll(-1))*segment).sum()/segment.sum().clamp_min(1e-12)
    li = mean_bp2/average_bp.square().clamp_min(1e-24)
    return torch.cat((axis,torch.stack((kappa,tritop,tribot,volume,li))))


def replay_frame(field, capture, r, z):
    """Exact forward at this field, local implicit-contour backward approximation."""
    field = field.double()
    axis = axis_position(field,capture["axis_cell"],r,z)
    if capture["points"] is None:
        zero = field.sum()*0
        return torch.cat((axis,zero.repeat(5))),None
    points = torch.as_tensor(capture["points"],device=field.device,dtype=field.dtype)
    support = torch.as_tensor(capture["support"],device=field.device,dtype=field.dtype)[None]
    values, gradient = sample(field,points,r,z,derivative=True)
    level = sample(field,support,r,z)[0]
    delta = (level-level.detach())-(values-values.detach())
    gradient = gradient.detach()
    floor = (field.detach().std(unbiased=False)/(r[-1]-r[0])*1e-4).square().clamp_min(1e-24)
    # Only the differential is applied: the forward contour is always exactly
    # the official one, with zero displacement at the captured field.
    points = points+delta[:,None]*gradient/gradient.square().sum(-1,keepdim=True).clamp_min(floor)
    inside = torch.as_tensor(capture["inside"],device=field.device)
    return contour_scalars(field,points,inside,r,z,axis),points


class ScorerGeometryLoss:
    """Supervision on official quantities, with training-only normalization.

    Exact forward agreement is tested separately from usefulness of approximate
    gradients. A failed prediction is penalized rather than removed from a batch.
    """
    def __init__(self,target,r,z,envelope,*,shot_names=None,workers=1,statistics=None,
                 population_counts=None, population_lcfs_frame_counts=None,
                 population_lcfs_mean_radius=None,contour_backward="implicit",
                 contour_temperature=.002):
        self.r,self.z = np.asarray(r,float),np.asarray(z,float)
        self.envelope = np.asarray(envelope,bool)
        self.workers = workers
        if contour_backward not in ("implicit", "predicted_level_residual",
                                    "symmetric_level_residual"):
            raise ValueError("unknown scorer contour backward")
        if contour_temperature <= 0 or not np.isfinite(contour_temperature):
            raise ValueError("contour temperature must be finite and positive")
        self.contour_backward = contour_backward
        self.contour_temperature = float(contour_temperature)
        self.target = target.detach().double()
        self.reference = self.capture(self.target)
        values = np.stack([item["scalars"] for item in self.reference])
        self.valid = np.isfinite(values)
        count = self.valid.sum(0)
        means = np.where(self.valid,values,0).sum(0)/np.maximum(count,1)
        variance = np.where(self.valid,(values-means)**2,0).sum(0)/np.maximum(count,1)
        if statistics is None:
            self.mean,self.std = means,np.maximum(variance,.001**2)**.5
        else:
            self.mean,self.std = map(np.asarray,statistics)
        if self.mean.shape != (7,) or self.std.shape != (7,) or not np.isfinite(self.std).all() or (self.std<=0).any():
            raise ValueError("invalid training-only consistency statistics")
        self.values = np.where(self.valid,values,self.mean)
        groups = np.arange(len(target)) if shot_names is None else np.asarray(shot_names)
        self.rgeo = np.ones(len(target))
        for name in np.unique(groups):
            rows = np.flatnonzero(groups == name)
            radii = [.5*(self.reference[i]["points"][:,0].max()+self.reference[i]["points"][:,0].min())
                     for i in rows if self.reference[i]["points"] is not None]
            if population_lcfs_mean_radius is not None and name in population_lcfs_mean_radius:
                self.rgeo[rows] = float(population_lcfs_mean_radius[name])
            else:
                self.rgeo[rows] = np.mean(radii) if radii else 1.
        self.shot_names = groups
        self.population_counts = (None if population_counts is None
                                  else np.asarray(population_counts, dtype=np.int64))
        if self.population_counts is not None and (self.population_counts.shape != (7,)
                                                    or (self.population_counts <= 0).any()):
            raise ValueError("population consistency counts must be seven positive integers")
        self.population_lcfs_frame_counts = (None if population_lcfs_frame_counts is None
                                             else dict(population_lcfs_frame_counts))
        if self.population_lcfs_frame_counts is not None and (
                not self.population_lcfs_frame_counts
                or any(value <= 0 for value in self.population_lcfs_frame_counts.values())):
            raise ValueError("population LCFS frame counts must be positive")

    def capture(self,field):
        fields = field.detach().cpu().double().numpy()
        def one(frame):
            return capture_frame(frame,self.r,self.z,self.envelope)
        if self.workers == 1:
            return list(map(one,fields))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one,fields))

    def __call__(self,field):
        if field.shape != self.target.shape:
            raise ValueError("geometry reference batch mismatch")
        captures = self.capture(field)
        r = torch.tensor(self.r,device=field.device,dtype=torch.float64)
        z = torch.tensor(self.z,device=field.device,dtype=torch.float64)
        scalar_rows, distances, distance_rows = [],[],[]
        level_distances, level_distance_rows = [],[]
        symmetric_level_distances = []
        for i,capture in enumerate(captures):
            values,points = replay_frame(field[i],capture,r,z)
            exact = torch.tensor(np.nan_to_num(capture["scalars"],nan=0.),device=field.device)
            # Correct harmless numerical replay differences in the forward only.
            values = exact+(values-values.detach())
            valid = torch.tensor(np.isfinite(capture["scalars"]),device=field.device)
            mean = torch.tensor(self.mean,device=field.device)
            scalar_rows.append(torch.where(valid,values,mean+field[i].sum()*0))
            target_points = self.reference[i]["points"]
            if target_points is None:
                continue  # reference has no scored LCFS, matching the scorer
            distance_rows.append(i)
            if points is None:
                distances.append(field[i].sum()*0+1.)
                level_distance_rows.append(i)
                level_distances.append(field[i].sum()*0+1.)
                symmetric_level_distances.append(field[i].sum()*0+1.)
                continue
            p_np = capture["points"]
            near_ref = cKDTree(target_points).query(p_np)[1]
            near_pred = cKDTree(p_np).query(target_points)[1]
            ref = torch.tensor(target_points,device=field.device,dtype=torch.float64)
            d1 = torch.linalg.vector_norm(points-ref[near_ref],dim=-1).max()
            d2 = torch.linalg.vector_norm(points[near_pred]-ref,dim=-1).max()
            distances.append((torch.maximum(d1,d2)/self.rgeo[i]).clamp_max(1.))
            level_distance_rows.append(i)
            target = torch.as_tensor(target_points,device=field.device,dtype=torch.float64)
            support = torch.as_tensor(capture["support"],device=field.device,
                                      dtype=torch.float64)[None]
            target_values, target_gradient = sample(field[i],target,r,z,derivative=True)
            separatrix_value = sample(field[i],support,r,z)[0]
            floor = (field[i].detach().std(unbiased=False)/(r[-1]-r[0])*1e-4).square().clamp_min(1e-24)
            normal_distance = ((target_values-separatrix_value).abs()
                               / target_gradient.detach().square().sum(-1).clamp_min(floor).sqrt()
                               / self.rgeo[i]).clamp_max(1.)
            temperature = self.contour_temperature
            level_distances.append(temperature*torch.logsumexp(normal_distance/temperature,dim=0))
            if self.contour_backward == "symmetric_level_residual":
                predicted_to_target = (
                    torch.linalg.vector_norm(points-ref[near_ref],dim=-1)
                    / self.rgeo[i]
                ).clamp_max(1.)
                both_directions = torch.cat((predicted_to_target,normal_distance))
                # Smooth the maximum over both directed Hausdorff residuals.
                # The forward score remains the exact scorer contour distance.
                symmetric_level_distances.append(
                    temperature*(torch.logsumexp(both_directions/temperature,dim=0)
                                 -np.log(len(both_directions)))
                )
        predictions = torch.stack(scalar_rows)
        ref = torch.tensor(self.values,device=field.device)
        std = torch.tensor(self.std,device=field.device)
        mask = torch.tensor(self.valid,device=field.device)
        errors = ((predictions-ref)/std).square()*mask
        if self.population_counts is None:
            scalar_denominator = mask.sum(0).clamp_min(1)
        else:
            scalar_denominator = torch.as_tensor(self.population_counts, device=field.device)
            if bool((scalar_denominator < mask.sum(0)).any()):
                raise ValueError("population consistency counts do not cover this batch")
        per_scalar = errors.sum(0)/scalar_denominator
        consistency = per_scalar[mask.any(0)].mean()
        # Each shot, not each frame, contributes equally to D_LCFS.
        if self.population_lcfs_frame_counts is None:
            shot_distances = []
            for name in np.unique(self.shot_names[distance_rows]):
                selected = [distance for row,distance in zip(distance_rows,distances) if self.shot_names[row] == name]
                shot_distances.append(torch.stack(selected).mean())
            contour = torch.stack(shot_distances).mean() if shot_distances else field.sum()*0
        else:
            contour = field.sum()*0
            n_shots = len(self.population_lcfs_frame_counts)
            for row,distance in zip(distance_rows,distances):
                name = self.shot_names[row]
                if name not in self.population_lcfs_frame_counts:
                    raise ValueError("LCFS row is outside the frozen fit population")
                contour = contour + distance/(n_shots*self.population_lcfs_frame_counts[name])
        level_contour = field.sum()*0
        if self.population_lcfs_frame_counts is None:
            level_shot_distances = []
            for name in np.unique(self.shot_names[level_distance_rows]):
                selected = [distance for row,distance in zip(level_distance_rows,level_distances)
                            if self.shot_names[row] == name]
                level_shot_distances.append(torch.stack(selected).mean())
            level_contour = (torch.stack(level_shot_distances).mean()
                             if level_shot_distances else field.sum()*0)
        else:
            n_shots = len(self.population_lcfs_frame_counts)
            for row,distance in zip(level_distance_rows,level_distances):
                name = self.shot_names[row]
                if name not in self.population_lcfs_frame_counts:
                    raise ValueError("level-set row is outside the frozen fit population")
                level_contour = level_contour + distance/(n_shots*self.population_lcfs_frame_counts[name])
        if self.contour_backward == "predicted_level_residual":
            contour = contour.detach() + (level_contour-level_contour.detach())
        elif self.contour_backward == "symmetric_level_residual":
            symmetric_level_contour = field.sum()*0
            if self.population_lcfs_frame_counts is None:
                symmetric_shot_distances = []
                for name in np.unique(self.shot_names[level_distance_rows]):
                    selected = [distance for row,distance in zip(
                        level_distance_rows,symmetric_level_distances)
                        if self.shot_names[row] == name]
                    symmetric_shot_distances.append(torch.stack(selected).mean())
                symmetric_level_contour = (
                    torch.stack(symmetric_shot_distances).mean()
                    if symmetric_shot_distances else field.sum()*0
                )
            else:
                n_shots = len(self.population_lcfs_frame_counts)
                for row,distance in zip(level_distance_rows,symmetric_level_distances):
                    name = self.shot_names[row]
                    if name not in self.population_lcfs_frame_counts:
                        raise ValueError("symmetric level-set row is outside the frozen fit population")
                    symmetric_level_contour = symmetric_level_contour + distance/(
                        n_shots*self.population_lcfs_frame_counts[name]
                    )
            contour = contour.detach() + (
                symmetric_level_contour-symmetric_level_contour.detach()
            )
        failure_fraction = float(np.mean([not np.isfinite(c["scalars"]).all() for c in captures]))
        if not torch.isfinite(consistency+contour):
            raise RuntimeError("non-finite scorer-geometry loss")
        return consistency,contour,dict(captures=captures,scalars=predictions,
            per_scalar=per_scalar,failure_fraction=failure_fraction,
            exact_contour=contour.detach(),predicted_level_residual=level_contour)
