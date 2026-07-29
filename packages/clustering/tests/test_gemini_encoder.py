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

"""Unit tests for the Gemini text-embedding encoder (litellm mocked — no key)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from clustering.config.schema import EncoderConfig
from clustering.encoders.gemini import GeminiEncoder


def _cfg(**extra: Any) -> EncoderConfig:
    return EncoderConfig(
        name="gemini",
        model_id="gemini/text-embedding-004",
        embedding_dim=3,
        extra={"api_key": "test-key", **extra},
    )


def test_gemini_encodes_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_embedding(*, model: str, input: list[str], api_key: str, timeout: float) -> Any:
        seen.append(list(input))
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2, 0.3]} for _ in input])

    monkeypatch.setattr("litellm.embedding", fake_embedding)
    enc = GeminiEncoder(_cfg(batch_size=2))
    out = enc.encode(["a", "b", "c"])

    assert out.pooled.shape == (3, 3)
    assert enc.multi_vector is False
    assert [len(b) for b in seen] == [2, 1]  # honoured batch_size=2


def test_gemini_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        GeminiEncoder(EncoderConfig(name="gemini", embedding_dim=3, extra={}))


def test_gemini_rejects_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embedding(*, model: str, input: list[str], api_key: str, timeout: float) -> Any:
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2]} for _ in input])  # 2-d ≠ cfg 3

    monkeypatch.setattr("litellm.embedding", fake_embedding)
    with pytest.raises(ValueError, match="embedding_dim"):
        GeminiEncoder(_cfg()).encode(["a"])
