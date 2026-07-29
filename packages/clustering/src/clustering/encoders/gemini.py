# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemini text-embedding encoder.

Embeds text with a Google Gemini embedding model through ``litellm`` — already
a dependency (used by the LLM clustering / classification paths), provider-
agnostic, so this adds no new package or license. The API key is read from the
encoder config ``extra``: a literal ``api_key``, or (preferred) an
``api_key_env`` naming the environment variable (default ``GEMINI_API_KEY``).

Single-vector text encoder. Set ``embedding_dim`` to the model's output width
(3072 for ``gemini/gemini-embedding-001``); a mismatch is caught at encode
time rather than corrupting downstream fusion silently.

Select with ``encoder_text=gemini``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder

_DEFAULT_MODEL = "gemini/gemini-embedding-001"


class GeminiEncoder(Encoder[str]):
    """Text embeddings from a Gemini model, via ``litellm.embedding``."""

    multi_vector = False

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        del device  # API-based: no local device
        self.embedding_dim = cfg.embedding_dim
        self.model = cfg.model_id or _DEFAULT_MODEL
        self.batch_size = max(1, int(cfg.extra.get("batch_size", 100)))
        self.timeout = float(cfg.extra.get("timeout", 60.0))
        key = cfg.extra.get("api_key")
        if not key:
            env = str(cfg.extra.get("api_key_env", "GEMINI_API_KEY"))
            key = os.environ.get(env)
            if not key:
                raise ValueError(
                    "Gemini encoder needs an API key: set encoder_text.extra.api_key or "
                    f"the ${env} environment variable."
                )
        self.api_key = str(key)

    @torch.no_grad()
    def encode(self, batch: Sequence[str]) -> EncoderOutput:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover — optional dep
            raise ImportError(
                "The Gemini encoder requires litellm. Install it with "
                "`pip install dgml-clustering[gemini]` (or `dgml[clustering]`, which "
                "already includes it)."
            ) from exc

        rows: list[list[float]] = []
        for start in range(0, len(batch), self.batch_size):
            chunk = [str(x) for x in batch[start : start + self.batch_size]]
            resp = litellm.embedding(
                model=self.model, input=chunk, api_key=self.api_key, timeout=self.timeout
            )
            for item in resp.data:
                vec = getattr(item, "embedding", None)
                if vec is None:
                    vec = item["embedding"]
                if len(vec) != self.embedding_dim:
                    raise ValueError(
                        f"Gemini model {self.model!r} returned {len(vec)}-d embeddings but "
                        f"encoder_text.embedding_dim={self.embedding_dim}; set embedding_dim "
                        "to match the model's output width."
                    )
                rows.append(list(vec))
        return EncoderOutput(pooled=torch.tensor(rows, dtype=torch.float32))


def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return GeminiEncoder(cfg, device=device)


register_encoder("gemini")(_factory)
