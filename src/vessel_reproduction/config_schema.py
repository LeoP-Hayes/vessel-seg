from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal


Taxonomy = Literal["algorithm", "engineering"]


@dataclass(frozen=True)
class MetaConfig:
    project: str = "vessel_reproduction"
    version: int = 1
    description: str = "Unsupervised multi-scale morphology vessel extraction"


@dataclass(frozen=True)
class PyramidConfig:
    scales: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])


@dataclass(frozen=True)
class MorphologyConfig:
    num_directions: int = 9
    line_length_per_scale: List[int] = field(default_factory=lambda: [6, 3, 2])


@dataclass(frozen=True)
class IlluminationConfig:
    mean_filter_size_per_scale: List[int] = field(default_factory=lambda: [7, 5, 3])
    epsilon_c: float = 1.0


@dataclass(frozen=True)
class ThresholdConfig:
    method: str = "adaptive_mean"
    adaptive_block_size: int = 21
    adaptive_C: float = -2.0


@dataclass(frozen=True)
class PostprocessConfig:
    area_min: int = 30


@dataclass(frozen=True)
class FusionConfig:
    method: str = "pixel_or"


@dataclass(frozen=True)
class PreprocessConfig:
    invert_intensity: bool = False


@dataclass(frozen=True)
class AlgorithmConfig:
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass(frozen=True)
class IOConfig:
    input_dir: str = "data"
    output_dir: str = "outputs"
    input_glob: str = "*.png"
    recursive: bool = False


@dataclass(frozen=True)
class OutputConfig:
    save_mask: bool = True
    save_overlay: bool = True
    save_intermediate: bool = False
    intermediate_stages: List[str] = field(
        default_factory=lambda: ["illumination", "tophat_response", "threshold", "postprocess"]
    )
    snapshot_filename: str = "run_config_snapshot.yaml"
    unsup_metrics_filename: str = "metrics_unsup.csv"


@dataclass(frozen=True)
class RuntimeConfig:
    num_workers: int = 0
    batch_size: int = 1
    device: str = "cpu"


@dataclass(frozen=True)
class RobustnessConfig:
    skip_on_error: bool = True
    max_errors: int = 50


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "run.log"


@dataclass(frozen=True)
class EngineeringConfig:
    io: IOConfig = field(default_factory=IOConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass(frozen=True)
class SnapshotConfig:
    include: List[str] = field(
        default_factory=lambda: [
            "meta",
            "algorithm",
            "engineering",
            "runtime_context",
            "param_taxonomy",
            "config_digest",
        ]
    )
    redact_keys: List[str] = field(default_factory=list)
    format: str = "yaml"


@dataclass(frozen=True)
class AppConfig:
    meta: MetaConfig = field(default_factory=MetaConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    engineering: EngineeringConfig = field(default_factory=EngineeringConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PARAM_TAXONOMY: Dict[str, Taxonomy] = {
    "algorithm.pyramid.scales": "algorithm",
    "algorithm.morphology.num_directions": "algorithm",
    "algorithm.morphology.line_length_per_scale": "algorithm",
    "algorithm.illumination.mean_filter_size_per_scale": "algorithm",
    "algorithm.illumination.epsilon_c": "algorithm",
    "algorithm.threshold.method": "algorithm",
    "algorithm.threshold.adaptive_block_size": "algorithm",
    "algorithm.threshold.adaptive_C": "algorithm",
    "algorithm.postprocess.area_min": "algorithm",
    "algorithm.fusion.method": "algorithm",
    "algorithm.preprocess.invert_intensity": "algorithm",
    "engineering.io.input_dir": "engineering",
    "engineering.io.output_dir": "engineering",
    "engineering.io.input_glob": "engineering",
    "engineering.io.recursive": "engineering",
    "engineering.output.save_mask": "engineering",
    "engineering.output.save_overlay": "engineering",
    "engineering.output.save_intermediate": "engineering",
    "engineering.output.intermediate_stages": "engineering",
    "engineering.output.snapshot_filename": "engineering",
    "engineering.output.unsup_metrics_filename": "engineering",
    "engineering.runtime.num_workers": "engineering",
    "engineering.runtime.batch_size": "engineering",
    "engineering.runtime.device": "engineering",
    "engineering.robustness.skip_on_error": "engineering",
    "engineering.robustness.max_errors": "engineering",
    "engineering.logging.level": "engineering",
    "engineering.logging.log_to_file": "engineering",
    "engineering.logging.log_filename": "engineering",
}
