"""Text towers (contract C6).

HashingTextEncoder is the offline default: word uni/bi-grams + character trigrams are hashed with a
stable CRC32 into an EmbeddingBag, then projected by a small MLP. Same text + same seed => same output.
SentenceTransformerTextEncoder is an optional adapter (import-guarded, tests skip when missing).
"""

from __future__ import annotations

import re
import zlib
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class TextEncoder(Protocol):
    text_dim: int

    def forward(self, texts: Sequence[str]) -> Tensor: ...

    def __call__(self, texts: Sequence[str]) -> Tensor: ...


def hashed_features(
    text: str,
    *,
    num_buckets: int,
    word_ngrams: tuple[int, ...] = (1, 2),
    char_ngrams: tuple[int, ...] = (3,),
) -> list[int]:
    tokens = _TOKEN_RE.findall(text.lower())
    feats: list[str] = []
    for n in word_ngrams:
        for i in range(len(tokens) - n + 1):
            feats.append(f"w{n}:{' '.join(tokens[i : i + n])}")
    for token in tokens:
        padded = f"<{token}>"
        for n in char_ngrams:
            if len(padded) < n:
                feats.append(f"c{n}:{padded}")
                continue
            for i in range(len(padded) - n + 1):
                feats.append(f"c{n}:{padded[i : i + n]}")
    if not feats:
        feats.append("<empty>")
    return [zlib.crc32(f.encode("utf-8")) % num_buckets for f in feats]


class HashingTextEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_buckets: int = 4096,
        text_dim: int = 64,
        hidden_dim: int | None = None,
        word_ngrams: tuple[int, ...] = (1, 2),
        char_ngrams: tuple[int, ...] = (3,),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.text_dim = int(text_dim)
        self.word_ngrams = tuple(word_ngrams)
        self.char_ngrams = tuple(char_ngrams)
        hidden = int(hidden_dim or text_dim * 2)
        generator = torch.Generator().manual_seed(int(seed))
        self.bag = nn.EmbeddingBag(self.num_buckets, self.text_dim, mode="mean")
        with torch.no_grad():
            self.bag.weight.copy_(torch.randn(self.bag.weight.shape, generator=generator) * 0.1)
        self.mlp = nn.Sequential(
            nn.Linear(self.text_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.text_dim),
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * (1.0 / layer.in_features**0.5))
                    layer.bias.zero_()
        self.norm = nn.LayerNorm(self.text_dim)
        self._cache: dict[str, list[int]] = {}

    def _features(self, text: str) -> list[int]:
        cached = self._cache.get(text)
        if cached is None:
            cached = hashed_features(
                text, num_buckets=self.num_buckets, word_ngrams=self.word_ngrams, char_ngrams=self.char_ngrams
            )
            self._cache[text] = cached
        return cached

    def forward(self, texts: Sequence[str]) -> Tensor:
        device = self.bag.weight.device
        if len(texts) == 0:
            return torch.zeros((0, self.text_dim), device=device)
        flat: list[int] = []
        offsets: list[int] = []
        for text in texts:
            offsets.append(len(flat))
            flat.extend(self._features(text))
        indices = torch.tensor(flat, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
        pooled = self.bag(indices, offset_tensor)
        return self.norm(pooled + self.mlp(pooled))


class SentenceTransformerTextEncoder(nn.Module):
    """Frozen sentence-transformers model + trainable linear projection to text_dim (optional extra)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", *, text_dim: int = 64, device: str | None = None) -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
            raise ImportError(
                "SentenceTransformerTextEncoder needs `sentence-transformers`, which is a declared "
                "dependency of this project. Run `uv sync --frozen`; if it is still missing, refresh the "
                "lock with the cooling-off rule in docs/checkpoints.md."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        base_dim = int(self.model.get_sentence_embedding_dimension())
        self.text_dim = int(text_dim)
        self.proj = nn.Linear(base_dim, self.text_dim)

    def forward(self, texts: Sequence[str]) -> Tensor:
        with torch.no_grad():
            base = self.model.encode(list(texts), convert_to_tensor=True, show_progress_bar=False)
        return self.proj(base.to(self.proj.weight.device).float())


class CachedSentenceTransformerTower(nn.Module):
    """Frozen sentence encoder + trainable projection, with the base embeddings memoised.

    The alignment loop re-encodes the same node and fact texts every step. Running a transformer
    over them each time would spend the training budget on the frozen part, so the base
    embeddings are computed once per unique string and cached; only the projection is trained.
    That keeps the compute budget comparable to the hashing tower it replaces (charter v2 §V2-C).
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        revision: str | None = None,
        text_dim: int = 64,
        device: str | None = None,
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        from .llm_bridge.hf_adapter import TEXT_TOWER_MODEL_ID, TEXT_TOWER_REVISION, load_text_tower

        self.model_id = model_id or TEXT_TOWER_MODEL_ID
        self.revision = revision or TEXT_TOWER_REVISION
        self.text_dim = int(text_dim)
        self.batch_size = int(batch_size)
        self._base = load_text_tower(model_id=self.model_id, revision=self.revision, device=device)
        get_dim = getattr(self._base, "get_embedding_dimension", None) or self._base.get_sentence_embedding_dimension
        base_dim = int(get_dim())
        self.base_dim = base_dim
        self.proj = nn.Linear(base_dim, self.text_dim)
        self._cache: dict[str, Tensor] = {}

    def _base_embeddings(self, texts: Sequence[str]) -> Tensor:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            with torch.no_grad():
                vectors = self._base.encode(
                    missing,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=self.batch_size,
                )
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = vector.detach().to("cpu")
        return torch.stack([self._cache[t] for t in texts])

    def forward(self, texts: Sequence[str]) -> Tensor:
        device = self.proj.weight.device
        if len(texts) == 0:
            return torch.zeros((0, self.text_dim), device=device)
        base = self._base_embeddings(list(texts)).to(device=device, dtype=self.proj.weight.dtype)
        return self.proj(base)


def build_text_encoder(kind: str, **kwargs) -> nn.Module:
    if kind == "hashing":
        return HashingTextEncoder(**kwargs)
    if kind == "sentence_transformer":
        return SentenceTransformerTextEncoder(**kwargs)
    if kind == "cached_sentence_transformer":
        return CachedSentenceTransformerTower(**kwargs)
    raise KeyError(
        f"unknown text encoder kind {kind!r}; expected 'hashing', 'sentence_transformer', "
        "or 'cached_sentence_transformer'"
    )
