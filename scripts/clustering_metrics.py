#!/usr/bin/env python3
"""Score a ``dgml cluster`` run from its JSON output — and diff two runs.

This reads the envelope the **shipped CLI actually wrote**, so it answers
questions about the product rather than about an in-process re-run:

    dgml cluster --workspace ./ws --json > before.json
    # ... turn on clustering.scenario.consolidation in the workspace config ...
    dgml cluster --workspace ./ws --json > after.json
    dgml file list --workspace ./ws --json > files.json

    uv run python scripts/clustering_metrics.py \
        --before before.json --after after.json --files files.json

It reports, per side:

* **Descriptive** — documents, #DocSets, singletons, largest cluster, review
  queue, mean/median/min assignment confidence. Straight from the envelope; no
  ground truth needed.
* **External** (vs. ground truth) — Adjusted Rand Index, Normalized / Adjusted
  Mutual Information, homogeneity, completeness, V-measure, purity, and mapped
  accuracy. The standard measures of *how well the discovered clusters recover
  the known categories*, computed by :mod:`clustering.metrics` — the same
  implementation the evaluation harness uses, not a private copy.

With ``--after`` it also lists every document that changed DocSet between the
two runs, which is the actual review artifact for a consolidation pass.

Ground truth comes from ``--labels`` (a ``{relative/path: label}`` map, or the
inverted ``{label: [path, ...]}`` form, matched on filename) or, failing that,
from the parent folder of each file's original path — the one-folder-per-class
corpus layout. Where the labels are ambiguous or absent the script says so
instead of quietly scoring against a degenerate truth.

Internal geometric metrics (silhouette, Davies-Bouldin) are intentionally
omitted: they need the document embeddings, which the CLI output does not carry.

It is not part of the public ``dgml`` CLI surface; it is an evaluation tool.

Usage:
    uv run python scripts/clustering_metrics.py --before cluster.json \
        --files files.json [--after cluster_after.json] [--labels labels.json]
        [--json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from clustering.metrics import external_metrics

# How many reassignments to print before collapsing into a count.
_MAX_LISTED_MOVES = 25


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _warn(message: str) -> None:
    print(f"  ! {message}", file=sys.stderr)


# ── Ground truth ─────────────────────────────────────────────────────────────


def _labels_by_basename(raw: Any) -> dict[str, str]:
    """``{basename: label}`` from either orientation of a labels JSON.

    Accepts ``{path: label}`` and the inverted ``{label: [path, ...]}`` — both
    are in use around the repo (``scripts/cluster_eval.py`` takes the latter),
    and guessing wrong would silently label nothing.

    A basename claimed by two different labels is **dropped**, not last-wins:
    with a one-folder-per-class layout the same ``contract.pdf`` can appear
    under two classes, and picking one at random corrupts every score derived
    from it.
    """
    if not isinstance(raw, dict):
        raise ValueError("labels JSON must be an object, not a bare list or scalar")

    claims: dict[str, set[str]] = {}
    for key, value in raw.items():
        if isinstance(value, str):  # {path: label}
            claims.setdefault(Path(key).name, set()).add(value)
        elif isinstance(value, list):  # {label: [path, ...]}
            for path in value:
                if isinstance(path, str):
                    claims.setdefault(Path(path).name, set()).add(str(key))
        else:
            raise ValueError(
                f"labels JSON value for {key!r} is neither a label string nor a list of paths"
            )

    resolved = {name: next(iter(labels)) for name, labels in claims.items() if len(labels) == 1}
    ambiguous = len(claims) - len(resolved)
    if ambiguous:
        _warn(
            f"{ambiguous} filename(s) in --labels are claimed by more than one label and were "
            "dropped from the ground truth"
        )
    return resolved


def build_truth(files: Iterable[dict[str, Any]], labels_path: str | None) -> dict[str, str]:
    """Map ``file_id -> ground-truth label``.

    Prefers the ``--labels`` map (matched on basename); falls back to the parent
    directory of each file's recorded original path. The fallback is reported
    when it is doing most of the work, because a flat corpus makes it produce a
    single pseudo-class for everything — which is exactly the case
    :func:`clustering.metrics.external_metrics` refuses to score.
    """
    by_basename: dict[str, str] = {}
    if labels_path:
        if not Path(labels_path).is_file():
            raise FileNotFoundError(f"--labels file not found: {labels_path}")
        by_basename = _labels_by_basename(_load(labels_path))

    truth: dict[str, str] = {}
    total = 0
    from_fallback = 0
    for rec in files:
        fid = rec.get("id")
        if not fid:
            continue
        total += 1
        original = rec.get("original_path") or rec.get("original_filename") or ""
        label = by_basename.get(Path(original).name)
        if label is None:
            # One-folder-per-class layout: the parent directory is the class.
            # A flat path has no parent, hence no derivable class — such a file
            # is left out of the truth rather than pooled into a pseudo-class,
            # which would look like a real class the clusterer failed to find.
            label = Path(original).parent.name
            from_fallback += 1
        if label:
            truth[fid] = label

    if from_fallback and by_basename:
        _warn(
            f"{from_fallback} of {total} file(s) were not in --labels; their class was taken "
            "from the parent folder"
        )
    if not truth:
        _warn(
            f"no ground-truth label could be resolved for any of the {total} file(s) — pass "
            "--labels, or use a corpus laid out one folder per class"
        )
    elif len(set(truth.values())) < 2:
        _warn(
            "the resolved ground truth has only one distinct class — external metrics will be "
            "reported as unavailable rather than as zeros"
        )
    return truth


# ── Per-run metrics ──────────────────────────────────────────────────────────


def _assignments(cluster: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The ``assignments`` map, keeping only entries that named a DocSet.

    An entry without a ``docset`` has nothing to score or count; the real
    envelope never emits one (unassigned files go to ``failed_file_ids``), so
    this only guards against hand-edited or older JSON.
    """
    raw = cluster.get("assignments") or {}
    if not isinstance(raw, dict):
        raise ValueError("cluster JSON 'assignments' must be an object")
    return {
        fid: det
        for fid, det in raw.items()
        if isinstance(det, dict) and det.get("docset") is not None
    }


def _aligned(
    assigns: dict[str, dict[str, Any]], truth: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Ground-truth and predicted labels, aligned over the docs present in both."""
    shared = [fid for fid in assigns if fid in truth]
    return [truth[fid] for fid in shared], [str(assigns[fid]["docset"]) for fid in shared]


def descriptive_metrics(cluster: dict[str, Any]) -> dict[str, Any]:
    """Shape and confidence stats derived straight from the cluster envelope."""
    assigns = _assignments(cluster)
    sizes: dict[str, int] = {}
    confs: list[float] = []
    flagged = 0
    for det in assigns.values():
        docset = str(det["docset"])
        sizes[docset] = sizes.get(docset, 0) + 1
        conf = det.get("confidence")
        if conf is not None:
            confs.append(float(conf))
        if det.get("review"):
            flagged += 1

    # Prefer the envelope's own list, but only when the key is actually present:
    # an empty list is a real answer ("nothing flagged"), not a missing one.
    queue = cluster.get("review_queue")
    if isinstance(queue, list):
        if len(queue) != flagged:
            _warn(
                f"review_queue has {len(queue)} entries but {flagged} assignment(s) carry the "
                "review flag — the two are derived from the same flags, so this JSON is stale"
            )
        review_queue = len(queue)
    else:
        review_queue = flagged

    size_vals = sorted(sizes.values(), reverse=True)
    return {
        "documents": len(assigns),
        "docsets": len(sizes),
        "new_docsets": cluster.get("n_new_clusters", 0),
        "assigned_existing": cluster.get("n_assigned_existing", 0),
        "singletons": sum(1 for v in size_vals if v == 1),
        "largest_cluster": size_vals[0] if size_vals else 0,
        "review_queue": review_queue,
        "failed": len(cluster.get("failed_file_ids") or []),
        "mean_conf": statistics.fmean(confs) if confs else None,
        "median_conf": statistics.median(confs) if confs else None,
        "min_conf": min(confs) if confs else None,
        "scored_conf": len(confs),
    }


def compute(cluster: dict[str, Any], truth: dict[str, str]) -> dict[str, Any]:
    """Descriptive plus external metrics for one run."""
    out = descriptive_metrics(cluster)
    gt, pred = _aligned(_assignments(cluster), truth)
    out.update(external_metrics(gt, pred).to_dict())
    return out


# ── Reporting ────────────────────────────────────────────────────────────────

# (label, key, kind) — a None key starts a new section.
_ROWS: list[tuple[str, str | None, str]] = [
    ("documents clustered", "documents", "int"),
    ("# DocSets (clusters)", "docsets", "int"),
    ("  new DocSets created", "new_docsets", "int"),
    ("  assigned to existing", "assigned_existing", "int"),
    ("singleton clusters", "singletons", "int"),
    ("largest cluster size", "largest_cluster", "int"),
    ("review queue size", "review_queue", "int"),
    ("failed / unclusterable", "failed", "int"),
    ("documents with a score", "scored_conf", "int"),
    ("mean confidence", "mean_conf", "float"),
    ("median confidence", "median_conf", "float"),
    ("min confidence", "min_conf", "float"),
    ("— external vs ground truth —", None, "section"),
    ("scored documents", "n_scored", "int"),
    ("# true classes", "n_true_classes", "int"),
    ("Adjusted Rand Index", "ari", "float"),
    ("Normalized MI", "nmi", "float"),
    ("Adjusted MI", "ami", "float"),
    ("homogeneity", "homogeneity", "float"),
    ("completeness", "completeness", "float"),
    ("V-measure", "v_measure", "float"),
    ("purity", "purity", "float"),
    ("mapped accuracy", "mapped_accuracy", "float"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def report(before: dict[str, Any], after: dict[str, Any] | None) -> None:
    label_w = max(len(lbl) for lbl, _, _ in _ROWS)
    single = after is None
    header = f"  {'metric'.ljust(label_w)}   {'value':>8}"
    rule = f"  {'-' * label_w}   {'-' * 8}"
    if not single:
        header = f"  {'metric'.ljust(label_w)}   {'before':>8}   {'after':>8}   delta"
        rule = f"{rule}   {'-' * 8}   -----"
    print(header)
    print(rule)

    for lbl, key, _kind in _ROWS:
        if key is None:
            print(f"\n  {lbl}")
            continue
        b = before.get(key)
        if single:
            print(f"  {lbl.ljust(label_w)}   {_fmt(b):>8}")
            continue
        assert after is not None
        a = after.get(key)
        delta = ""
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            delta = (
                f"{a - b:+.3f}" if isinstance(b, float) or isinstance(a, float) else f"{a - b:+d}"
            )
        print(f"  {lbl.ljust(label_w)}   {_fmt(b):>8}   {_fmt(a):>8}   {delta}")

    for side, metrics in (("before", before), ("after", after)):
        note = (metrics or {}).get("note")
        if note:
            print(f"\n  no external scores for '{side}': {note}")


def reassignments(before_json: dict[str, Any], after_json: dict[str, Any]) -> None:
    """List the documents that changed DocSet between the two runs."""
    ba = _assignments(before_json)
    aa = _assignments(after_json)
    shared = ba.keys() & aa.keys()
    moved = [
        (fid, ba[fid]["docset"], aa[fid]["docset"])
        for fid in sorted(shared)
        if ba[fid]["docset"] != aa[fid]["docset"]
    ]
    print(f"\n  Reassignments: {len(moved)} of {len(shared)} shared document(s) changed DocSet")
    for fid, from_ds, to_ds in moved[:_MAX_LISTED_MOVES]:
        print(f"    {fid}:  {from_ds}  →  {to_ds}")
    if len(moved) > _MAX_LISTED_MOVES:
        print(f"    … and {len(moved) - _MAX_LISTED_MOVES} more")

    # A document assigned in only one of the two runs is not a reassignment, but
    # it does mean the two sides are not scored over the same set — worth saying
    # out loud before anyone reads a delta as an improvement.
    only_before = len(ba.keys() - aa.keys())
    only_after = len(aa.keys() - ba.keys())
    if only_before or only_after:
        print(
            f"    ({only_before} assigned only in 'before', {only_after} only in 'after' — "
            "the two runs did not place the same documents)"
        )


def _file_records(raw: Any) -> list[dict[str, Any]]:
    """File records from a ``dgml file list`` envelope or a bare list of them."""
    records = raw.get("files") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError(
            "--files must be `dgml file list --json` output (an object with a 'files' array) "
            "or a bare array of file records"
        )
    return [rec for rec in records if isinstance(rec, dict)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--before", required=True, help="cluster JSON (the 'before' / only run).")
    p.add_argument("--after", default=None, help="cluster JSON from the second run, to diff.")
    p.add_argument("--files", required=True, help="`dgml file list --json` output.")
    p.add_argument("--labels", default=None, help="Optional {relpath: label} JSON map.")
    p.add_argument("--json", action="store_true", help="Also dump the raw metric dicts as JSON.")
    args = p.parse_args(argv)

    truth = build_truth(_file_records(_load(args.files)), args.labels)

    before_json = _load(args.before)
    after_json = _load(args.after) if args.after else None
    before = compute(before_json, truth)
    after = compute(after_json, truth) if after_json is not None else None

    report(before, after)
    if after_json is not None:
        reassignments(before_json, after_json)

    if args.json:
        print("\n--- raw JSON ---")
        print(json.dumps({"before": before, "after": after}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
