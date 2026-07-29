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
agnostic, so this adds no new package or license.

Single-vector text encoder. Set ``embedding_dim`` to the model's output width
(3072 for ``gemini/gemini-embedding-001``); a mismatch is caught at encode
time rather than corrupting downstream fusion silently.

Three things separate an API encoder from the local ones, and each is a knob
here rather than an assumption:

* **Credentials.** Read from ``extra.api_key``, else the env var named by
  ``extra.api_key_env`` (default ``GEMINI_API_KEY``). Finding neither is *not*
  an error: ``api_key`` stays ``None`` and litellm resolves it from its own
  environment, which is how Vertex service-account and ``GOOGLE_API_KEY``
  setups work. A genuinely absent credential surfaces as litellm's own
  authentication error on the first call.
* **Input length.** Gemini embedding models cap input at a couple of thousand
  tokens and reject anything longer, so a full multi-page document does not
  fit. ``cfg.max_length`` is honoured as a *token* budget (the same unit the
  local encoders use) and applied by truncating the text, since there is no
  local tokenizer to count with — see :data:`_CHARS_PER_TOKEN` for why that is
  an approximation and not a guarantee. ``None`` sends the text through
  untouched, which is correct when the caller already truncated.
* **Failure.** One rate-limit response would otherwise discard every embedding
  already paid for in the same run, so requests retry (``extra.num_retries``,
  default :data:`_DEFAULT_NUM_RETRIES`).

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

# Retries are per-request inside litellm and cover 429 / 5xx / timeout. Three
# is enough to ride out the burst rate limits an embedding sweep provokes
# without turning a hard failure into a long hang.
_DEFAULT_NUM_RETRIES = 3

# ``cfg.max_length`` is a token count; the API takes text. Converting needs a
# chars-per-token figure, and 4 is the usual English average — an *average*,
# not an upper bound. A token-dense document (dense tables, long numbers, many
# rare words) can still exceed the model's cap after truncation, in which case
# the API rejects the request and the fix is a lower ``max_length``. Erring
# high rather than low is deliberate: silently discarding text a model would
# have accepted costs recall on exactly the long documents where the extra
# pages carry the signal.
_CHARS_PER_TOKEN = 4


class GeminiEncoder(Encoder[str]):
    """Text embeddings from a Gemini model, via ``litellm.embedding``."""

    multi_vector = False

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        del device  # API-based: no local device
        self.embedding_dim = cfg.embedding_dim
        self.model = cfg.model_id or _DEFAULT_MODEL
        self.batch_size = max(1, int(cfg.extra.get("batch_size", 100)))
        self.timeout = float(cfg.extra.get("timeout", 60.0))
        self.num_retries = max(0, int(cfg.extra.get("num_retries", _DEFAULT_NUM_RETRIES)))
        self.max_chars = (
            None if cfg.max_length is None else max(1, int(cfg.max_length) * _CHARS_PER_TOKEN)
        )
        # Optional passthroughs. Only sent when the caller sets them, so the
        # provider's own defaults (and any measurement taken against them)
        # stay in force. ``dimensions`` asks the model to truncate its output
        # via Matryoshka representation learning; it has to agree with
        # ``embedding_dim``, which the width check below enforces anyway.
        self.extra_kwargs: dict[str, Any] = {}
        for key in ("task_type", "dimensions"):
            value = cfg.extra.get(key)
            if value is not None:
                self.extra_kwargs[key] = value
        api_key: Any = cfg.extra.get("api_key")
        if not api_key:
            env = str(cfg.extra.get("api_key_env", "GEMINI_API_KEY"))
            api_key = os.environ.get(env)
        # ``None`` on purpose when nothing was found — litellm then applies its
        # own credential resolution (GOOGLE_API_KEY, Vertex ADC, …) instead of
        # us pre-emptively failing a setup that would have worked.
        self.api_key: str | None = str(api_key) if api_key else None

    def _prepare(self, batch: Sequence[str]) -> list[str]:
        texts = [str(x) for x in batch]
        if self.max_chars is None:
            return texts
        return [t[: self.max_chars] for t in texts]

    @torch.no_grad()
    def encode(self, batch: Sequence[str]) -> EncoderOutput:
        # An empty batch reaches here from CachingEncoder (a fully-cached
        # batch forwards the empty remainder). Answer it with a correctly
        # shaped empty tensor instead of spending a request: ``torch.tensor([])``
        # would be 1-D and break the caller's ``[N, D]`` contract.
        if not batch:
            return EncoderOutput(pooled=torch.zeros((0, self.embedding_dim), dtype=torch.float32))

        try:
            import litellm
        except ImportError as exc:  # pragma: no cover — optional dep
            raise ImportError(
                "The Gemini encoder requires litellm. Install it with "
                "`pip install dgml-clustering[gemini]` (or `dgml[clustering]`, which "
                "already includes it)."
            ) from exc

        texts = self._prepare(batch)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            resp = litellm.embedding(
                model=self.model,
                input=chunk,
                api_key=self.api_key,
                timeout=self.timeout,
                num_retries=self.num_retries,
                **self.extra_kwargs,
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
        if len(rows) != len(texts):
            # Row order carries the doc↔vector alignment for the whole
            # pipeline. A short (or long) response would silently misattribute
            # every embedding after the gap.
            raise ValueError(
                f"Gemini returned {len(rows)} embedding(s) for {len(texts)} input(s); "
                "the response cannot be aligned back to the batch."
            )
        return EncoderOutput(pooled=torch.tensor(rows, dtype=torch.float32))


def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return GeminiEncoder(cfg, device=device)


register_encoder("gemini")(_factory)
