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

"""External cluster-validity metrics — does a partition recover known classes?

Pure label-list math. Every function here takes two equal-length sequences of
label strings — ground truth and prediction — and returns numbers. Nothing in
this module knows about scenarios, embeddings, encoders, or the ``dgml cluster``
JSON envelope, so the same implementation serves a scenario test, the evaluation
harness, and a post-hoc scoring script instead of each growing its own copy.

Only *external* metrics live here: the ones that compare a partition against a
known labeling. Internal geometric scores (silhouette, Davies-Bouldin) need the
embedding vectors rather than just the labels, so they belong wherever the
vectors are.

Two of the metrics answer the same question differently, and the difference
matters when reading a report:

- :func:`purity` asks "within each predicted cluster, how big is the majority
  true class?". Several clusters may claim the same class, so splitting one
  class across many clusters costs nothing — in the limit, one document per
  cluster scores a perfect 1.0.
- :func:`mapped_accuracy` forces a one-to-one cluster→class assignment (the
  Hungarian algorithm), so over-clustering is penalised. It is the more honest
  single number whenever the cluster count is not pinned to the class count,
  which for emergent clustering is always.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

# Below this many distinct ground-truth classes there is nothing to recover: a
# single-class truth makes completeness and purity trivially 1.0 and drives ARI
# and the mutual-information scores to ~0 regardless of how good the partition
# is. Reporting those numbers would read as "the clustering failed" when what
# actually happened is "the labels carry no information".
_MIN_TRUE_CLASSES = 2

_NOTE_EMPTY = "no documents to score"
_NOTE_SINGLE_CLASS = (
    "ground truth has fewer than 2 distinct classes, so external scores are "
    "not meaningful and were not computed"
)


@dataclass(frozen=True)
class ExternalMetrics:
    """Standard external scores, or the reason there are none.

    Every score is ``None`` together, and ``note`` says why, whenever the inputs
    are degenerate (see :data:`_MIN_TRUE_CLASSES`). The counts are always
    populated — knowing that 40 documents were scored against 1 class is the
    diagnosis, so it is reported even when nothing else can be.
    """

    n_scored: int
    n_true_classes: int
    n_pred_clusters: int
    ari: float | None = None
    nmi: float | None = None
    ami: float | None = None
    homogeneity: float | None = None
    completeness: float | None = None
    v_measure: float | None = None
    purity: float | None = None
    mapped_accuracy: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """A plain dict, suitable for JSON output or a report table."""
        return asdict(self)


def _checked(true_labels: Sequence[str], pred_labels: Sequence[str]) -> int:
    """Length of the aligned pair, raising when the caller mis-aligned them."""
    if len(true_labels) != len(pred_labels):
        raise ValueError(
            f"true_labels and pred_labels must be the same length, "
            f"got {len(true_labels)} and {len(pred_labels)}"
        )
    return len(true_labels)


def purity(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    """Fraction of documents in the majority true class of their own cluster.

    ``0.0`` for an empty input. Note that this rewards over-clustering; pair it
    with :func:`mapped_accuracy` before drawing a conclusion.
    """
    n = _checked(true_labels, pred_labels)
    if n == 0:
        return 0.0
    by_cluster: dict[str, Counter[str]] = defaultdict(Counter)
    for truth, pred in zip(true_labels, pred_labels, strict=True):
        by_cluster[pred][truth] += 1
    correct = sum(counts.most_common(1)[0][1] for counts in by_cluster.values())
    return correct / n


def mapped_accuracy(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    """Accuracy under the best **one-to-one** cluster→class mapping.

    The mapping is the one that maximises the number of correctly-placed
    documents, found with the Hungarian algorithm over the contingency table.
    Because no two clusters may claim the same class, this does not reward
    splitting a class across clusters the way :func:`purity` does.

    Clusters left unmatched (there are more clusters than classes, or vice
    versa) contribute nothing, so the score is bounded by
    ``min(n_classes, n_clusters) / n`` in the worst case. ``0.0`` for an empty
    input.
    """
    n = _checked(true_labels, pred_labels)
    if n == 0:
        return 0.0

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    classes = sorted(set(true_labels))
    clusters = sorted(set(pred_labels))
    class_index = {name: i for i, name in enumerate(classes)}
    cluster_index = {name: i for i, name in enumerate(clusters)}

    contingency = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for truth, pred in zip(true_labels, pred_labels, strict=True):
        contingency[cluster_index[pred], class_index[truth]] += 1

    # linear_sum_assignment minimises, so negate to maximise the matched count.
    rows, cols = linear_sum_assignment(-contingency)
    return int(contingency[rows, cols].sum()) / n


def external_metrics(true_labels: Sequence[str], pred_labels: Sequence[str]) -> ExternalMetrics:
    """The full external suite for one aligned (truth, prediction) pair.

    Degenerate inputs come back with every score ``None`` and a ``note``
    explaining which one it was, rather than a plausible-looking zero.
    """
    n = _checked(true_labels, pred_labels)
    n_true = len(set(true_labels))
    n_pred = len(set(pred_labels))

    if n == 0:
        return ExternalMetrics(n_scored=0, n_true_classes=0, n_pred_clusters=0, note=_NOTE_EMPTY)
    if n_true < _MIN_TRUE_CLASSES:
        return ExternalMetrics(
            n_scored=n,
            n_true_classes=n_true,
            n_pred_clusters=n_pred,
            note=_NOTE_SINGLE_CLASS,
        )

    from sklearn import metrics as skmetrics

    homogeneity, completeness, v_measure = skmetrics.homogeneity_completeness_v_measure(
        true_labels, pred_labels
    )
    return ExternalMetrics(
        n_scored=n,
        n_true_classes=n_true,
        n_pred_clusters=n_pred,
        ari=float(skmetrics.adjusted_rand_score(true_labels, pred_labels)),
        nmi=float(skmetrics.normalized_mutual_info_score(true_labels, pred_labels)),
        ami=float(skmetrics.adjusted_mutual_info_score(true_labels, pred_labels)),
        homogeneity=float(homogeneity),
        completeness=float(completeness),
        v_measure=float(v_measure),
        purity=purity(true_labels, pred_labels),
        mapped_accuracy=mapped_accuracy(true_labels, pred_labels),
    )
