"""BirdCLEF+ 2026 experiment configuration."""

from __future__ import annotations

from pathlib import Path

_ON_KAGGLE = Path("/kaggle/input").exists()

if _ON_KAGGLE:
    _DATA_ROOT = "/kaggle/input/birdclef-2026"
else:
    _DATA_ROOT = "/home/rise/Documents/Acoustics/BirdCLEF/birdclef-2026"

from dataclasses import dataclass, field, replace
_OUTPUT_DIR = _DATA_ROOT + "/experiments"
_EXPERIMENT_NAME = "exp030_pcen_fold0"
_EXPERIMENT_DIR = _OUTPUT_DIR + "/" + _EXPERIMENT_NAME


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 32000
    duration: int = 5
    n_samples: int = 160000


@dataclass(frozen=True)
class MelConfig:
    n_mels: int = 256
    fmin: int = 250
    fmax: int = 16000
    hop_length: int = 512
    n_fft: int = 2048
    mel_scale: str = "htk"
    use_pcen: bool = True
    pcen_time_constant: float = 0.4
    pcen_eps: float = 1e-6
    pcen_gain: float = 0.98
    pcen_bias: float = 2.0
    pcen_power: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 40
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    n_folds: int = 5
    train_folds: tuple[int, ...] = (1, 2, 3, 4)
    val_fold: int = 0
    early_stopping_patience: int = 8
    experiment_name: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "tf_efficientnet_b5_ns"
    pretrained: bool = True
    num_classes: int = 234
    gem_p: float = 3.0
    dropout: float = 0.2


@dataclass(frozen=True)
class PathsConfig:
    data_root: str = _DATA_ROOT
    train_audio: str = _DATA_ROOT + "/train_audio"
    train_soundscapes: str = _DATA_ROOT + "/train_soundscapes"
    train_csv: str = _DATA_ROOT + "/train.csv"
    taxonomy_csv: str = _DATA_ROOT + "/taxonomy.csv"
    soundscape_labels: str = _DATA_ROOT + "/train_soundscapes_labels.csv"
    perch_onnx: str = _DATA_ROOT + "/perch_onnx/perch_v2.onnx"
    output_dir: str = _OUTPUT_DIR
    experiment_name: str = _EXPERIMENT_NAME
    experiment_dir: str = _EXPERIMENT_DIR
    checkpoints_dir: str = _EXPERIMENT_DIR + "/checkpoints"
    logs_dir: str = _EXPERIMENT_DIR + "/logs"
    oof_dir: str = _EXPERIMENT_DIR + "/oof"
    submission_dir: str = _EXPERIMENT_DIR + "/submission"


@dataclass(frozen=True)
class AugmentationConfig:
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    use_specaugment: bool = True
    freq_mask_param: int = 30
    time_mask_param: int = 50
    use_noise_mix: bool = True
    noise_mix_prob: float = 1.0
    noise_mix_alpha_min: float = 0.4
    noise_mix_alpha_max: float = 0.7
    use_gain_aug: bool = False
    gain_aug_prob: float = 0.5
    gain_min: float = 0.3
    gain_max: float = 2.0


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


config = Config()


def paths_with_experiment_name(paths: PathsConfig, experiment_name: str) -> PathsConfig:
    """Return a new ``PathsConfig`` with ``experiment_dir`` and subpaths under ``experiment_name``."""
    name = experiment_name.strip()
    ed = paths.output_dir + "/" + name
    return replace(
        paths,
        experiment_name=name,
        experiment_dir=ed,
        checkpoints_dir=ed + "/checkpoints",
        logs_dir=ed + "/logs",
        oof_dir=ed + "/oof",
        submission_dir=ed + "/submission",
    )


def apply_experiment_name_override(experiment_name: str) -> None:
    """
    Mutate module-level ``config`` so checkpoints/logs use ``experiment_name``.

    Sets ``config.training.experiment_name`` and rebuilds ``config.paths`` experiment fields.
    """
    global config
    name = experiment_name.strip()
    if not name:
        return
    new_paths = paths_with_experiment_name(config.paths, name)
    new_training = replace(config.training, experiment_name=name)
    config = replace(config, paths=new_paths, training=new_training)


def create_experiment_dirs(paths: PathsConfig | None = None) -> None:
    """Create experiment_dir and subfolders: checkpoints, logs, oof, submission."""
    p = paths if paths is not None else config.paths
    root = Path(p.experiment_dir)
    for name in ("checkpoints", "logs", "oof", "submission"):
        (root / name).mkdir(parents=True, exist_ok=True)
