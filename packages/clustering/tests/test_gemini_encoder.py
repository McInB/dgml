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
import torch
from clustering.config.schema import EncoderConfig
from clustering.encoders.caching import encoder_fingerprint
from clustering.encoders.gemini import GeminiEncoder

MODEL = "gemini/gemini-embedding-001"


def _cfg(*, max_length: int | None = None, **extra: Any) -> EncoderConfig:
    return EncoderConfig(
        name="gemini",
        model_id=MODEL,
        embedding_dim=3,
        max_length=max_length,
        extra={"api_key": "test-key", **extra},
    )


def _fake_embedding(recorder: list[dict[str, Any]]) -> Any:
    """A litellm.embedding stub whose vectors identify their own input.

    Row order is the only thing tying a document to its vector, so the stub
    encodes the input's length into the vector: a response reordered or
    shortened by the encoder shows up as a wrong row, not as a shape that
    happens to match.
    """

    def fake(*, model: str, input: list[str], **kwargs: Any) -> Any:
        recorder.append({"model": model, "input": list(input), **kwargs})
        return SimpleNamespace(data=[{"embedding": [float(len(t)), 0.0, 1.0]} for t in input])

    return fake


def test_gemini_encodes_in_batches_preserving_row_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    enc = GeminiEncoder(_cfg(batch_size=2))
    out = enc.encode(["a", "bb", "ccc", "dddd", "eeeee"])

    assert out.pooled.shape == (5, 3)
    assert enc.multi_vector is False
    # Batched 2 + 2 + 1, and every row sits where its document was.
    assert [c["input"] for c in calls] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert out.pooled[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_gemini_truncates_to_max_length(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    # 10 tokens -> 40 characters. Gemini rejects over-long input outright, so
    # an untruncated multi-page document would fail the whole request.
    enc = GeminiEncoder(_cfg(max_length=10))
    out = enc.encode(["x" * 500, "short"])

    assert [len(t) for t in calls[0]["input"]] == [40, 5]
    assert out.pooled[:, 0].tolist() == [40.0, 5.0]


def test_gemini_without_max_length_sends_text_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    GeminiEncoder(_cfg()).encode(["y" * 500])

    assert [len(t) for t in calls[0]["input"]] == [500]


def test_gemini_retries_and_passes_optional_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    GeminiEncoder(_cfg()).encode(["a"])
    # Retries default on: one 429 must not discard a run's paid-for embeddings.
    assert calls[0]["num_retries"] == 3
    # Provider defaults left alone unless asked for.
    assert "task_type" not in calls[0] and "dimensions" not in calls[0]

    calls.clear()
    GeminiEncoder(_cfg(num_retries=0, task_type="CLUSTERING", dimensions=3)).encode(["a"])
    assert calls[0]["num_retries"] == 0
    assert calls[0]["task_type"] == "CLUSTERING"
    assert calls[0]["dimensions"] == 3


def test_gemini_empty_batch_costs_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    # A fully-cached batch forwards its empty remainder here; the result still
    # has to be [0, D] so the caller can stack it.
    out = GeminiEncoder(_cfg()).encode([])

    assert out.pooled.shape == (0, 3)
    assert calls == []


def test_gemini_rejects_a_short_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*, model: str, input: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(data=[{"embedding": [1.0, 2.0, 3.0]}])  # 1 row for 2 inputs

    monkeypatch.setattr("litellm.embedding", fake)
    with pytest.raises(ValueError, match="cannot be aligned"):
        GeminiEncoder(_cfg()).encode(["a", "b"])


def test_gemini_without_a_key_defers_to_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    # No key in extra and none in the env is not an error: litellm resolves
    # credentials itself (GOOGLE_API_KEY, Vertex ADC, ...), so failing here
    # would break setups that work.
    enc = GeminiEncoder(EncoderConfig(name="gemini", model_id=MODEL, embedding_dim=3, extra={}))
    assert enc.api_key is None
    enc.encode(["a"])
    assert calls[0]["api_key"] is None


def test_gemini_reads_the_key_from_the_named_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_GEMINI_KEY", "from-env")
    cfg = EncoderConfig(
        name="gemini",
        model_id=MODEL,
        embedding_dim=3,
        extra={"api_key_env": "MY_GEMINI_KEY"},
    )
    assert GeminiEncoder(cfg).api_key == "from-env"


def test_gemini_rejects_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*, model: str, input: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(data=[{"embedding": [0.1, 0.2]} for _ in input])  # 2-d ≠ cfg 3

    monkeypatch.setattr("litellm.embedding", fake)
    with pytest.raises(ValueError, match="embedding_dim"):
        GeminiEncoder(_cfg()).encode(["a"])


def test_gemini_accepts_object_style_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # litellm returns objects for some providers and dicts for others.
    def fake(*, model: str, input: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 0.5, 1.0]) for _ in input])

    monkeypatch.setattr("litellm.embedding", fake)
    out = GeminiEncoder(_cfg()).encode(["a"])
    assert torch.allclose(out.pooled, torch.tensor([[0.0, 0.5, 1.0]]))


def test_api_key_does_not_namespace_the_embedding_cache() -> None:
    """Rotating a key must not orphan the vectors it already paid for."""

    def cfg(**extra: Any) -> EncoderConfig:
        return EncoderConfig(name="gemini", model_id=MODEL, embedding_dim=3, extra=extra)

    key_a = cfg(batch_size=4, api_key="aaa")
    assert encoder_fingerprint(key_a) == encoder_fingerprint(cfg(batch_size=4, api_key="bbb"))
    # Dropping the secret is a no-op for a config that has none, so caches
    # already on disk keep the fingerprint they were written under.
    assert encoder_fingerprint(key_a) == encoder_fingerprint(cfg(batch_size=4))
    # Everything else in `extra` still separates namespaces.
    assert encoder_fingerprint(key_a) != encoder_fingerprint(cfg(batch_size=8, api_key="aaa"))
