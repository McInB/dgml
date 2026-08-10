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

"""Assertions that compare two backends — the ones a single-backend run cannot make.

Everything else in this package can be checked one store at a time. These cannot:
the assertion *is* a comparison, so both results have to exist in the same
expression. Running a suite twice and eyeballing the output does not do it, and
persisting a value between runs is fragile in exactly the ways CI is.

The headline is the attestation root. DGML's proof-of-origin claim is that a
Merkle root identifies a file's content — if that root depends on where the
workspace happens to be hosted, the claim is not true.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.consistency import check_workspace
from dgml_core.docsets import DocSetStore
from dgml_core.file_attestation import attest_file, collect_file_version
from dgml_core.files import FileStore
from dgml_core.pages import GS_BINARIES
from dgml_core.storage import Workspace
from dgml_core.workspace_ops import WorkspaceOps

needs_gs = pytest.mark.skipif(
    not any(shutil.which(name) for name in GS_BINARIES), reason="ghostscript not installed"
)


def _ingest(ws: Workspace, source: Path) -> str:
    return FileStore(ws).add(source).record.id


@needs_gs
def test_identical_content_yields_an_identical_attestation_root(
    local_workspace: Workspace,
    s3_workspace: Workspace,
    make_pdf: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """The claim the whole storage layer has to preserve.

    Same bytes in, same proof out — whether the workspace lives on a laptop or
    in a bucket. This is the assertion that caught the ``staged_write``
    additiveness bug in reverse: stale page images on one backend changed the
    leaf set, and the roots diverged."""
    source = make_pdf(tmp_path / "contract.pdf", pages=3)

    local_id = _ingest(local_workspace, source)
    s3_id = _ingest(s3_workspace, source)

    local_attestation = attest_file(local_workspace, local_id)
    s3_attestation = attest_file(s3_workspace, s3_id)

    assert [a.slot_id for a in local_attestation.leaves] == [
        a.slot_id for a in s3_attestation.leaves
    ]
    assert [a.leaf_hash for a in local_attestation.leaves] == [
        a.leaf_hash for a in s3_attestation.leaves
    ]
    assert local_attestation.root == s3_attestation.root


@needs_gs
def test_a_shrinking_re_render_agrees_across_backends(
    local_workspace: Workspace,
    s3_workspace: Workspace,
    make_pdf: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Re-render a file with fewer pages on both backends.

    An additive ``staged_write`` leaves the surplus images behind on whichever
    store stages through temp files, and since ``collect_file_version`` hashes
    ``list_blobs(page_images)`` the roots would then differ. This is the
    regression guard for W2, now against a genuinely remote store rather than a
    test double."""
    source = make_pdf(tmp_path / "contract.pdf", pages=5)
    ids = {
        "local": _ingest(local_workspace, source),
        "s3": _ingest(s3_workspace, source),
    }

    roots = {}
    for label, ws in (("local", local_workspace), ("s3", s3_workspace)):
        file_id = ids[label]
        with ws.store.staged_write(layout.file_pages_prefix(file_id)) as pages_dir:
            for n in (1, 2):
                (pages_dir / f"page_{n}.png").write_bytes(f"page-{n}".encode())
        slots = [a.slot_id for a in collect_file_version(ws, file_id).artifacts]
        assert slots == ["source", "page_image[1]", "page_image[2]"], (label, slots)
        roots[label] = attest_file(ws, file_id).root

    assert roots["local"] == roots["s3"]


@needs_gs
def test_the_same_writes_leave_the_same_key_set(
    local_workspace: Workspace,
    s3_workspace: Workspace,
    make_pdf: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Identical operations must leave identical *addresses*, not merely
    equivalent content — otherwise a workspace could not be moved between
    backends, and ``dgml check`` would disagree with itself."""
    source = make_pdf(tmp_path / "contract.pdf", pages=2)
    keys = {}
    for label, ws in (("local", local_workspace), ("s3", s3_workspace)):
        file_id = _ingest(ws, source)
        docset_id = DocSetStore(ws).create(name="Invoices").id
        DocSetStore(ws).add_file(docset_id, file_id)
        ws.store.put_blob(layout.dgml_xml_key(docset_id, file_id, "contract"), b"<x/>")
        # Normalise the generated ids, which differ per workspace by design.
        keys[label] = sorted(
            k.replace(file_id, "<FILE>").replace(docset_id, "<DOCSET>")
            for k in ws.store.list_blobs("")
        )

    assert keys["local"] == keys["s3"]


@needs_gs
def test_a_cascade_leaves_the_same_end_state(
    local_workspace: Workspace,
    s3_workspace: Workspace,
    make_pdf: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """Delete the same thing on both backends and compare what survives."""
    source = make_pdf(tmp_path / "contract.pdf", pages=2)
    remaining = {}
    for label, ws in (("local", local_workspace), ("s3", s3_workspace)):
        file_id = _ingest(ws, source)
        docset_id = DocSetStore(ws).create(name="Invoices").id
        DocSetStore(ws).add_file(docset_id, file_id)
        ws.store.put_blob(layout.dgml_xml_key(docset_id, file_id, "contract"), b"<x/>")

        WorkspaceOps(ws).delete_file(file_id)

        remaining[label] = {
            "blobs": sorted(k.replace(docset_id, "<DOCSET>") for k in ws.store.list_blobs("")),
            "assignments": ws.store.find_docs(layout.Collection.ASSIGNMENTS, {}),
            "files": ws.store.find_docs(layout.Collection.FILES, {}),
            "issues": [i.kind for i in check_workspace(ws).issues],
        }

    assert remaining["local"] == remaining["s3"]
    assert remaining["s3"]["blobs"] == []
    assert remaining["s3"]["issues"] == []


def test_listings_are_ordered_identically_across_backends(
    local_workspace: Workspace,
    s3_workspace: Workspace,
    make_pdf: Callable[..., Path],
    tmp_path: Path,
) -> None:
    """``find_docs`` has no defined ordering — path order on local disk,
    insertion order in a document database. Anything user-visible must not
    inherit that, so the listings sort. Inserted deliberately out of order."""
    listings = {}
    for label, ws in (("local", local_workspace), ("s3", s3_workspace)):
        docsets = DocSetStore(ws)
        created = [docsets.create(name=n).id for n in ("Third", "First", "Second")]
        listings[label] = [d.id for d in docsets.list_all()]
        assert listings[label] == sorted(created), (label, listings[label])

    # Same ids cannot be compared across workspaces, but the *ordering rule* can:
    # both must be sorted, which is what makes the CLI output backend-independent.
    assert listings["local"] == sorted(listings["local"])
    assert listings["s3"] == sorted(listings["s3"])
