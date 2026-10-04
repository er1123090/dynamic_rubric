from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


class LocalEmbeddingError(RuntimeError):
    pass


class LocalBGEEmbeddingProvider:
    """Pinned BGE-M3 dense embedding provider using normalized CLS vectors."""

    def __init__(
        self,
        model_path: Path,
        model: str,
        revision: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model_path = model_path
        self.model_id = model
        self.revision = revision
        self.device = device
        self.batch_size = batch_size
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def identity(self) -> Mapping[str, str]:
        return {
            "model": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "pooling": "normalized_cls",
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_dir():
            raise LocalEmbeddingError(f"embedding snapshot is absent: {self.model_path}")
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
        ).to(self.device)
        self._model.eval()

    def preflight(self) -> Mapping[str, Any]:
        vectors = self.embed(("criterion identity probe", "independent semantic probe"))
        return {**self.identity, "dimension": len(vectors[0]), "probe_count": len(vectors)}

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return ()
        self._load()
        import torch  # pyright: ignore[reportMissingImports]
        import torch.nn.functional as functional  # pyright: ignore[reportMissingImports]

        rows: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                current = list(texts[start : start + self.batch_size])
                encoded = self._tokenizer(
                    current,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                hidden = self._model(**encoded).last_hidden_state[:, 0].float()
                normalized = functional.normalize(hidden, p=2, dim=1)
                rows.extend(normalized.cpu().tolist())
        if len(rows) != len(texts) or any(not row for row in rows):
            raise LocalEmbeddingError("embedding result count or dimension mismatch")
        return rows
