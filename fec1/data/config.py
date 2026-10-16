"""The three preprocessing settings consumed by the FEC1 input pipeline."""
from dataclasses import asdict, dataclass


@dataclass
class PreprocessingConfig:
    grid_size: int = 65
    max_tokens: int = 192
    token_features: int = 5

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(
            **{key: value for key, value in values.items() if key in cls.__dataclass_fields__}
        )
