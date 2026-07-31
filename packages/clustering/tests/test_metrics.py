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

"""External cluster-validity metrics."""

from __future__ import annotations

import pytest
from clustering.metrics import ExternalMetrics, external_metrics, mapped_accuracy, purity

# ── purity ───────────────────────────────────────────────────────────────────


def test_purity_is_one_for_a_perfect_partition() -> None:
    truth = ["a", "a", "b", "b"]
    pred = ["c0", "c0", "c1", "c1"]
    assert purity(truth, pred) == 1.0


def test_purity_is_label_name_agnostic() -> None:
    # A partition is only ever as good as its grouping; the names the clusterer
    # invents ("unknown_0") carry no information and must not be compared to the
    # ground-truth names.
    truth = ["invoice", "invoice", "contract"]
    assert purity(truth, ["unknown_7", "unknown_7", "unknown_3"]) == 1.0


def test_purity_counts_only_the_majority_of_each_cluster() -> None:
    truth = ["a", "a", "a", "b"]
    pred = ["c0", "c0", "c0", "c0"]  # one cluster: majority "a" = 3 of 4
    assert purity(truth, pred) == 0.75


def test_purity_rewards_over_clustering() -> None:
    # This is the documented weakness that motivates mapped_accuracy: one
    # document per cluster is a useless partition that scores perfectly.
    truth = ["a", "a", "b", "b"]
    pred = ["c0", "c1", "c2", "c3"]
    assert purity(truth, pred) == 1.0


def test_purity_of_nothing_is_zero() -> None:
    assert purity([], []) == 0.0


# ── mapped_accuracy ──────────────────────────────────────────────────────────


def test_mapped_accuracy_is_one_for_a_perfect_partition() -> None:
    truth = ["a", "a", "b", "b"]
    pred = ["c1", "c1", "c0", "c0"]  # permuted names, same grouping
    assert mapped_accuracy(truth, pred) == 1.0


def test_mapped_accuracy_penalises_over_clustering_where_purity_does_not() -> None:
    truth = ["a", "a", "b", "b"]
    pred = ["c0", "c1", "c2", "c3"]
    # Only two clusters can be matched one-to-one to the two classes, so at most
    # 2 of the 4 documents count as correctly placed.
    assert purity(truth, pred) == 1.0
    assert mapped_accuracy(truth, pred) == 0.5


def test_mapped_accuracy_picks_the_globally_best_mapping() -> None:
    # Greedy per-cluster majority voting would assign both c0 and c1 to "a"
    # (c0: 2a/1b, c1: 2a/2b) and then have nothing left for "b". The Hungarian
    # assignment takes the 1-point loss on c0 to win 2 points on c1.
    truth = ["a", "a", "b", "a", "a", "b", "b"]
    pred = ["c0", "c0", "c0", "c1", "c1", "c1", "c1"]
    assert mapped_accuracy(truth, pred) == pytest.approx(4 / 7)


def test_mapped_accuracy_handles_fewer_clusters_than_classes() -> None:
    truth = ["a", "b", "c"]
    pred = ["c0", "c0", "c0"]
    # One cluster can claim one class, so exactly one document is placed.
    assert mapped_accuracy(truth, pred) == pytest.approx(1 / 3)


def test_mapped_accuracy_of_nothing_is_zero() -> None:
    assert mapped_accuracy([], []) == 0.0


# ── external_metrics ─────────────────────────────────────────────────────────


def test_external_metrics_scores_a_perfect_partition() -> None:
    truth = ["a", "a", "b", "b", "c", "c"]
    pred = ["c0", "c0", "c1", "c1", "c2", "c2"]
    m = external_metrics(truth, pred)
    assert m.n_scored == 6
    assert m.n_true_classes == 3
    assert m.n_pred_clusters == 3
    assert m.note is None
    assert m.ari == pytest.approx(1.0)
    assert m.nmi == pytest.approx(1.0)
    assert m.ami == pytest.approx(1.0)
    assert m.homogeneity == pytest.approx(1.0)
    assert m.completeness == pytest.approx(1.0)
    assert m.v_measure == pytest.approx(1.0)
    assert m.purity == pytest.approx(1.0)
    assert m.mapped_accuracy == pytest.approx(1.0)


def test_external_metrics_separates_homogeneity_from_completeness() -> None:
    # Splitting each true class in two is perfectly homogeneous (no cluster
    # mixes classes) but incomplete (a class is spread over two clusters).
    truth = ["a", "a", "b", "b"]
    pred = ["c0", "c1", "c2", "c3"]
    m = external_metrics(truth, pred)
    assert m.homogeneity == pytest.approx(1.0)
    assert m.completeness is not None
    assert m.completeness < 1.0


def test_external_metrics_of_nothing_reports_why() -> None:
    m = external_metrics([], [])
    assert m == ExternalMetrics(
        n_scored=0, n_true_classes=0, n_pred_clusters=0, note="no documents to score"
    )


def test_a_single_class_truth_is_reported_as_unscoreable_not_as_zero() -> None:
    # The load-bearing guard. A flat corpus (or a ground-truth map that failed
    # to resolve) yields one "class" for everything; sklearn would hand back
    # ari=0.0 / nmi=0.0, which reads as a failed clustering rather than as
    # useless labels.
    truth = ["a"] * 5
    pred = ["c0", "c0", "c1", "c1", "c2"]
    m = external_metrics(truth, pred)
    assert m.n_scored == 5
    assert m.n_true_classes == 1
    assert m.n_pred_clusters == 3
    assert m.note is not None and "fewer than 2 distinct classes" in m.note
    assert m.ari is None
    assert m.nmi is None
    assert m.purity is None
    assert m.mapped_accuracy is None


def test_a_single_predicted_cluster_is_still_scored() -> None:
    # Unlike a single-class truth, a single predicted cluster is a real and
    # informative failure — the near-zero ARI is a true statement about the run.
    truth = ["a", "a", "b", "b"]
    pred = ["c0"] * 4
    m = external_metrics(truth, pred)
    assert m.note is None
    assert m.ari == pytest.approx(0.0)
    assert m.completeness == pytest.approx(1.0)
    assert m.homogeneity == pytest.approx(0.0)


def test_mismatched_lengths_raise_rather_than_scoring_a_prefix() -> None:
    with pytest.raises(ValueError, match="same length"):
        external_metrics(["a", "b"], ["c0"])
    with pytest.raises(ValueError, match="same length"):
        purity(["a", "b"], ["c0"])
    with pytest.raises(ValueError, match="same length"):
        mapped_accuracy(["a", "b"], ["c0"])


def test_to_dict_round_trips_every_field() -> None:
    m = external_metrics(["a", "a", "b"], ["c0", "c0", "c1"])
    d = m.to_dict()
    assert d["n_scored"] == 3
    assert d["mapped_accuracy"] == pytest.approx(1.0)
    assert set(d) == {
        "n_scored",
        "n_true_classes",
        "n_pred_clusters",
        "ari",
        "nmi",
        "ami",
        "homogeneity",
        "completeness",
        "v_measure",
        "purity",
        "mapped_accuracy",
        "note",
    }
