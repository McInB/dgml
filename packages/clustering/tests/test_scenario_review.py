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

"""The ``review`` flag across S1-S5.

Every scenario now reports one boolean per document alongside its prediction.
The invariants these tests defend are the same in all five:

- **Off by default.** An unconfigured run flags nothing and its predictions are
  identical to what it produced before ``calibration`` existed.
- **Advisory, never a veto.** Turning the review gate all the way up must not
  move a single label. Routing to the ``unknown`` bucket is the *novelty*
  decision and belongs to ``threshold*``; review is a separate axis.
- **Only real assignments can be reviewed.** A document in the unknown bucket
  (or S1 noise) has no assignment for a human to confirm — it is waiting on a
  new category, which is a different question.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from clustering.config.schema import Config
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.encoders.base import Encoder, EncoderOutput
from clustering.scenarios import build_scenario
from clustering.scenarios.base import ScenarioResult
from PIL import Image


@dataclass(frozen=True)
class _Record:
    doc_id: str
    text: str
    label: str | None = None


class _InMemoryDataset(DocumentDataset):
    def __init__(self, records: list[_Record]) -> None:
        self._records = records
        self._image = Image.new("RGB", (8, 8))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> DocumentRecord:
        record = self._records[index]
        return DocumentRecord(
            doc_id=record.doc_id,
            label=record.label,
            image=self._image,
            text=record.text,
            thumbnail_path=None,
        )


class _LookupEncoder(Encoder[Any]):
    """Explicit 2-D vectors so the prototype geometry is readable by eye.

    Anything not in the table encodes to the origin. The table includes the
    category-name prompts S2/S4 build, so their name prototypes land in the same
    two places S3/S5's support means do — which is what makes one geometry
    reusable across all five scenarios.
    """

    embedding_dim = 2
    multi_vector = False

    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self._vectors = vectors

    def encode(self, batch: Sequence[Any]) -> EncoderOutput:
        rows = [self._vectors.get(str(item), (0.0, 0.0)) for item in batch]
        return EncoderOutput(pooled=torch.tensor(rows, dtype=torch.float32))


# ── Fixtures: one geometry, reused across scenarios ────────────────────────
# Two tight support classes at x=0 and x=10, plus three queries: one clearly in
# each class and one dead centre. The centre document is the interesting one —
# it should be assigned (it has a nearest prototype like anything else) and
# flagged (the assignment is a coin flip).
_VECTORS: dict[str, tuple[float, float]] = {
    "invoice_a": (0.0, 0.0),
    "invoice_b": (0.0, 1.0),
    "contract_a": (10.0, 0.0),
    "contract_b": (10.0, 1.0),
    "clear_invoice": (0.2, 0.5),
    "borderline": (5.0, 0.5),
    "clear_contract": (9.8, 0.5),
    # S2/S4 encode these prompts instead of samples; put their prototypes where
    # the support means are so the two paths share one geometry.
    "a scanned document of category: Invoice": (0.0, 0.5),
    "a scanned document of category: Contract": (10.0, 0.5),
}

_CATEGORIES = ["Invoice", "Contract"]


def _support() -> _InMemoryDataset:
    return _InMemoryDataset(
        [
            _Record("si1", "invoice_a", "Invoice"),
            _Record("si2", "invoice_b", "Invoice"),
            _Record("sc1", "contract_a", "Contract"),
            _Record("sc2", "contract_b", "Contract"),
        ]
    )


def _queries() -> _InMemoryDataset:
    return _InMemoryDataset(
        [
            _Record("clear_i", "clear_invoice", "Invoice"),
            _Record("mid", "borderline", "Invoice"),
            _Record("clear_c", "clear_contract", "Contract"),
        ]
    )


def _config(scenario_name: str, calibration: dict[str, Any] | None = None) -> Config:
    scenario: dict[str, Any] = {"name": scenario_name}
    if scenario_name == "s1":
        scenario["k_clusters"] = 2
    else:
        scenario["known_categories"] = list(_CATEGORIES)
    if scenario_name in ("s3", "s5"):
        scenario["n_shots"] = 2
    if calibration is not None:
        scenario["calibration"] = calibration
    return Config.model_validate(
        {
            "scenario": scenario,
            "encoder_text": {"name": "dummy", "embedding_dim": 2},
            "encoder_image": {"name": "dummy", "embedding_dim": 2},
            "fusion": {"name": "none", "prefer_modality": "text", "output_dim": 2},
            "manifold": {"name": "euclidean", "dim": 2, "curvature": 0.0},
            "training": {"epochs": 0, "identity_projector": True, "batch_size": 8},
            "logger": {"name": "none"},
            "corpus": {"root": "."},
            "device": "cpu",
            "seed": 0,
        }
    )


def _run(scenario_name: str, calibration: dict[str, Any] | None = None) -> ScenarioResult:
    scenario = build_scenario(_config(scenario_name, calibration))
    scenario.text_encoder = _LookupEncoder(_VECTORS)
    support = _support() if scenario_name in ("s3", "s5") else None
    return scenario.fit_predict(_queries(), support)


_ALL_SCENARIOS = ["s1", "s2", "s3", "s4", "s5"]


# ── Off by default ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", _ALL_SCENARIOS)
def test_review_is_empty_by_default(name: str) -> None:
    result = _run(name)

    # One flag per document, all clear: an unconfigured run never asks for
    # review, so nothing downstream changes until someone opts in.
    assert len(result.review) == len(result.doc_ids)
    assert not any(result.review)
    assert result.metadata["n_review"] == 0


@pytest.mark.parametrize("name", _ALL_SCENARIOS)
def test_review_flags_do_not_change_predictions(name: str) -> None:
    plain = _run(name)
    # A floor of 1.0 flags literally everything — the most aggressive gate
    # available. Not one label may move as a result.
    gated = _run(name, {"abstain_threshold": 1.0})

    assert gated.predictions == plain.predictions
    assert gated.confidence == plain.confidence


# ── The floor actually fires, and on the right documents ───────────────────
@pytest.mark.parametrize("name", ["s2", "s3", "s4", "s5"])
def test_abstain_floor_flags_everything_at_one(name: str) -> None:
    result = _run(name, {"abstain_threshold": 1.0})

    # Only documents that got a known-category assignment can be flagged; the
    # unknown bucket (S2/S3) is a novelty finding, not a review item.
    for pred, flag in zip(result.predictions, result.review, strict=True):
        assert flag == (pred in _CATEGORIES)
    assert result.metadata["n_review"] == sum(result.review)


@pytest.mark.parametrize("name", ["s4", "s5"])
def test_borderline_document_is_flagged_before_the_clear_ones(name: str) -> None:
    # S4/S5 are closed-set, so all three queries are assigned and the ranking is
    # visible. A floor between the borderline confidence and the clear ones must
    # pick out exactly the middle document.
    baseline = _run(name)
    confidences = [c for c in baseline.confidence if c is not None]
    assert len(confidences) == 3
    mid_conf = confidences[1]
    assert mid_conf < min(confidences[0], confidences[2])

    floor = (mid_conf + min(confidences[0], confidences[2])) / 2
    result = _run(name, {"abstain_threshold": floor})

    assert result.review == [False, True, False]
    assert result.predictions == baseline.predictions  # still assigned


def test_s1_review_excludes_noise_documents() -> None:
    # S1 pins noise to 0.0 confidence, which any floor would catch. Noise means
    # "no cluster fit this" — a novelty finding a reviewer cannot correct by
    # confirming an assignment — so it must stay out of the queue.
    scenario = build_scenario(_config("s1", {"abstain_threshold": 1.0}))
    scenario.text_encoder = _LookupEncoder(_VECTORS)
    result = scenario.fit_predict(_queries())

    for pred, flag in zip(result.predictions, result.review, strict=True):
        if pred == "cluster_noise":
            assert flag is False


# ── Calibration provenance ─────────────────────────────────────────────────
def test_s5_records_the_calibrator_it_fitted() -> None:
    result = _run("s5", {"method": "temperature", "coverage": 0.8})

    # The calibrated number is only interpretable with the operating point that
    # produced it, so the fit is echoed into metadata rather than discarded.
    calibration = result.metadata["calibration"]
    assert calibration is not None
    assert calibration["method"] == "temperature"
    assert calibration["coverage"] == 0.8
    assert calibration["n_calibration"] == 4  # one leave-one-out row per support doc


def test_s3_records_the_calibrator_it_fitted() -> None:
    result = _run("s3", {"method": "platt", "coverage": 0.9})

    calibration = result.metadata["calibration"]
    assert calibration is not None
    assert calibration["method"] == "platt"
    assert calibration["n_calibration"] == 4


@pytest.mark.parametrize("name", ["s2", "s4"])
def test_name_only_scenarios_never_fit_a_calibrator(name: str) -> None:
    # S2/S4 build prototypes from category *names*, so there is no labeled
    # support set to fit against. Asking for a method must not fabricate one:
    # the confidence stays ordinal and only the plain floor applies.
    result = _run(name, {"method": "temperature", "coverage": 0.9, "abstain_threshold": 1.0})

    assert result.metadata.get("calibration") is None
    assert any(result.review)  # the floor still works


def test_s5_calibration_changes_the_reported_confidence() -> None:
    plain = _run("s5")
    calibrated = _run("s5", {"method": "temperature"})

    # Temperature scaling rescales the confidence without reordering it: the
    # numbers move, the ranking does not, and the labels certainly do not.
    assert calibrated.predictions == plain.predictions
    plain_conf = [c for c in plain.confidence if c is not None]
    cal_conf = [c for c in calibrated.confidence if c is not None]
    assert cal_conf != plain_conf
    assert [i for i, _ in sorted(enumerate(cal_conf), key=lambda kv: kv[1])] == [
        i for i, _ in sorted(enumerate(plain_conf), key=lambda kv: kv[1])
    ]
