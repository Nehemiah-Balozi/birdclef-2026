"""
PERCH ONNX feature extractor (frozen, CPU) + trainable linear head on GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
except ImportError as e:
    ort = None  # type: ignore[assignment]
    _ORT_IMPORT_ERROR = e
else:
    _ORT_IMPORT_ERROR = None

from config import config


class PerchModel(nn.Module):
    """
    Run PERCH v2 ONNX on raw waveform ``[batch, 160000]`` float32 (CPU), then
    apply a trainable head on GPU: Linear(1536→512) → BN → ReLU → Dropout → Linear(512→234).
    """

    def __init__(
        self,
        onnx_path: str | Path | None = None,
        head_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if ort is None:
            raise ImportError(
                "onnxruntime is required for PerchModel. pip install onnxruntime"
            ) from _ORT_IMPORT_ERROR

        path = Path(onnx_path or config.paths.perch_onnx)
        if not path.is_file():
            raise FileNotFoundError(f"PERCH ONNX not found: {path}")

        self._head_device = head_device or torch.device("cuda")
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        in_meta = self._session.get_inputs()[0]
        self._input_name = in_meta.name

        n_cls = config.model.num_classes
        d = config.model.dropout
        self.head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=d),
            nn.Linear(512, n_cls),
        )
        self.head.to(self._head_device)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform :
            ``(batch, 160000)`` float32 on any device; PERCH runs on CPU.
        """
        if waveform.dim() != 2 or waveform.shape[1] != config.audio.n_samples:
            raise ValueError(
                f"Expected waveform (B, {config.audio.n_samples}), got {tuple(waveform.shape)}"
            )
        w = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
        outputs = self._session.run(None, {self._input_name: w})
        emb = np.asarray(outputs[0], dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != 1536:
            raise RuntimeError(
                f"Expected PERCH embedding (B, 1536), got {emb.shape}"
            )
        x = torch.from_numpy(emb).to(self._head_device, non_blocking=True)
        return self.head(x)


def _filter_head_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Strip outer prefix so keys match ``nn.Sequential`` state inside ``self.head``
    (``0.weight``, ``1.weight``, …).

    Accepted checkpoint prefixes:

    - ``head.*`` / ``module.head.*`` (same layout as :class:`PerchModel`)
    - ``net.*`` / ``module.net.*`` (PERCH head checkpoints: ``net.0.weight``, …)
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module.head."):
            out[k[len("module.head.") :]] = v
        elif k.startswith("module.net."):
            out[k[len("module.net.") :]] = v
        elif k.startswith("head."):
            out[k[len("head.") :]] = v
        elif k.startswith("net."):
            out[k[len("net.") :]] = v
    return out


class PerchTeacher(nn.Module):
    """
    Frozen PERCH ONNX (CPU) + frozen trained head (GPU): returns teacher logits (234).

    Head weights may be stored as ``head.*``, ``net.*``, or DDP-prefixed ``module.head.*`` /
    ``module.net.*``; see :func:`_filter_head_state_dict`.
    """

    def __init__(
        self,
        head_checkpoint: str | Path,
        onnx_path: str | Path | None = None,
        head_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if ort is None:
            raise ImportError("onnxruntime is required. pip install onnxruntime") from _ORT_IMPORT_ERROR

        path = Path(onnx_path or config.paths.perch_onnx)
        if not path.is_file():
            raise FileNotFoundError(f"PERCH ONNX not found: {path}")

        self._head_device = head_device or torch.device("cuda")
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        n_cls = config.model.num_classes
        d = config.model.dropout
        self.head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=d),
            nn.Linear(512, n_cls),
        )
        self.head.to(self._head_device)

        ckpt_path = Path(head_checkpoint)
        try:
            ckpt = torch.load(ckpt_path, map_location=self._head_device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self._head_device)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        head_sd = _filter_head_state_dict(state)
        if not head_sd:
            raise RuntimeError(
                f"No head weights (head.* / net.* / module.head.* / module.net.*) in checkpoint: {ckpt_path}"
            )
        self.head.load_state_dict(head_sd, strict=True)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """``waveform`` (B, n_samples) float32 on any device → teacher logits on head device."""
        if waveform.dim() != 2 or waveform.shape[1] != config.audio.n_samples:
            raise ValueError(
                f"Expected waveform (B, {config.audio.n_samples}), got {tuple(waveform.shape)}"
            )
        w = waveform.detach().cpu().numpy().astype(np.float32, copy=False)
        emb = np.asarray(self._session.run(None, {self._input_name: w})[0], dtype=np.float32)
        x = torch.from_numpy(emb).to(self._head_device, non_blocking=True)
        return self.head(x)
