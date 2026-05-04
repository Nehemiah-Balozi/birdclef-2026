"""BirdCLEF+ 2026 experiment configuration."""

from __future__ import annotations

from pathlib import Path

_ON_KAGGLE = Path("/kaggle/input").exists()

if _ON_KAGGLE:
    _DATA_ROOT = "/kaggle/input/birdclef-2026"
else:
    _DATA_ROOT = "/home/rise/Documents/Acoustics/BirdCLEF/birdclef-2026"

from dataclasses import dataclass, field
_OUTPUT_DIR = _DATA_ROOT + "/experiments"
_EXPERIMENT_NAME = "exp009_soundscape_mixed_fold2"
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


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


config = Config()


def create_experiment_dirs(paths: PathsConfig | None = None) -> None:
    """Create experiment_dir and subfolders: checkpoints, logs, oof, submission."""
    p = paths if paths is not None else config.paths
    root = Path(p.experiment_dir)
    for name in ("checkpoints", "logs", "oof", "submission"):
        (root / name).mkdir(parents=True, exist_ok=True)
