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

"""Real DGML workflows run against the sample store.

These are the tests that justify the package. Contract tests (``test_contract``)
prove the store obeys the interface — but the worst defect found while hardening
the storage layer was not a store bug at all: ``docsets/<id>/schema.json`` was
written with ``put_blob`` and read with ``get_doc``. Both methods worked
perfectly. The *caller* was wrong, and only a backend where blobs and documents
are genuinely different systems can reveal it.

So these drive the public API — ``FileStore``, ``DocSetStore``,
``check_workspace``, attestation — and never mention a filesystem path, which
makes them backend-agnostic by construction.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.consistency import check_workspace
from dgml_core.docsets import DocSetStore
from dgml_core.file_attestation import (
    attest_file,
    collect_file_version,
    export_attestation,
    verify_bundle,
)
from dgml_core.files import FileStore
from dgml_core.generation.rnc import write_docset_rnc
from dgml_core.pages import GS_BINARIES
from dgml_core.storage import Workspace
from dgml_core.usage import UsageEvent, read_events, record_usage
from dgml_core.workspace_ops import WorkspaceOps
from dgml_storage_s3 import S3MongoStore

needs_gs = pytest.mark.skipif(
    not any(shutil.which(name) for name in GS_BINARIES), reason="ghostscript not installed"
)


def _add_file(
    ws: Workspace,
    make_pdf: Callable[..., Path],
    tmp_path: Path,
    name: str = "doc.pdf",
    pages: int = 2,
) -> str:
    return FileStore(ws).add(make_pdf(tmp_path / name, pages)).record.id


# ---------------------------------------------------------------- the wiring


def test_workspace_resolves_the_sample_store(s3_workspace: Workspace) -> None:
    """Everything below depends on this: the pipeline is talking to S3 + Mongo,
    not quietly falling back to local disk."""
    assert isinstance(s3_workspace.store, S3MongoStore)


# ------------------------------------------------------------------ ingestion


@needs_gs
def test_file_add_stores_source_pages_and_text(
    s3_workspace: Workspace, make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    """Exercises the path bridges on a real backend for the first time.

    ``materialize`` hands ghostscript a real PDF path, ``staged_write`` collects
    the rendered pages and the extracted per-page text. On ``LocalStore`` both
    are zero-copy passthroughs; here they must actually round-trip through the
    object store."""
    file_id = _add_file(s3_workspace, make_pdf, tmp_path, pages=3)
    store = s3_workspace.store

    record = store.get_doc(layout.Collection.FILES, file_id)
    assert record is not None and record["page_count"] == 3
    assert store.blob_exists(layout.file_source_key(file_id, "doc.pdf"))
    assert len(store.list_blobs(layout.file_pages_prefix(file_id))) == 3
    assert len(store.list_blobs(layout.file_text_prefix(file_id))) == 3


@needs_gs
def test_check_is_clean_after_add(
    s3_workspace: Workspace, make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    """``dgml check`` walks the whole workspace through the store — manifests via
    ``find_docs``, artifacts via ``list_blobs``, hashes via ``sha256_blob``."""
    _add_file(s3_workspace, make_pdf, tmp_path)
    report = check_workspace(s3_workspace)
    assert report.issues == [], [i.to_json() for i in report.issues]
    assert report.files_checked == 1


@needs_gs
def test_re_render_replaces_stale_pages(
    s3_workspace: Workspace, make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    """The ``staged_write`` replace contract, on a backend that cannot cheat.

    Locally the renderer deletes stale ``page_*.png`` in place; through an object
    store the batch must be replaced explicitly, or a shrinking re-render leaves
    phantom pages that attestation would then hash."""
    file_id = _add_file(s3_workspace, make_pdf, tmp_path, pages=5)
    assert len(s3_workspace.store.list_blobs(layout.file_pages_prefix(file_id))) == 5

    with s3_workspace.store.staged_write(layout.file_pages_prefix(file_id)) as pages_dir:
        for n in (1, 2):
            (pages_dir / f"page_{n}.png").write_bytes(b"fake")

    assert len(s3_workspace.store.list_blobs(layout.file_pages_prefix(file_id))) == 2


# -------------------------------------------------------- docsets + cascades


@needs_gs
def test_assign_then_cascade_delete(
    s3_workspace: Workspace, make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    file_id = _add_file(s3_workspace, make_pdf, tmp_path)
    docsets = DocSetStore(s3_workspace)
    docset_id = docsets.create(name="Invoices").id
    docsets.add_file(docset_id, file_id)
    assert docsets.list_files(docset_id) == [file_id]

    # a generated artifact and its sidecar, so the cascade has all three kinds
    store = s3_workspace.store
    store.put_blob(layout.dgml_xml_key(docset_id, file_id, "doc"), b"<x/>")
    store.put_doc(
        layout.Collection.EXTRACTION_STATS, layout.pair_id(docset_id, file_id), {"matched": 1}
    )

    WorkspaceOps(s3_workspace).unassign(docset_id, file_id)

    assert docsets.list_files(docset_id) == []
    assert store.list_blobs(layout.docset_pair_prefix(docset_id, file_id)) == []
    assert (
        store.get_doc(layout.Collection.EXTRACTION_STATS, layout.pair_id(docset_id, file_id))
        is None
    )
    # the file itself survives — a docset is a grouping, not an owner
    assert store.get_doc(layout.Collection.FILES, file_id) is not None


@needs_gs
def test_delete_file_removes_everything_it_owns(
    s3_workspace: Workspace, make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    file_id = _add_file(s3_workspace, make_pdf, tmp_path)
    docsets = DocSetStore(s3_workspace)
    docset_id = docsets.create(name="Invoices").id
    docsets.add_file(docset_id, file_id)

    FileStore(s3_workspace).delete(file_id)

    store = s3_workspace.store
    assert store.get_doc(layout.Collection.FILES, file_id) is None
    assert store.list_blobs(layout.file_prefix(file_id)) == []
    assert store.find_docs(layout.Collection.ASSIGNMENTS, {"file_id": file_id}) == []
    assert check_workspace(s3_workspace).issues == []


# ------------------------------------------------- the schema.json regression


_SCHEMA = {
    "tags": {
        "Invoice": {
            "name": "Invoice",
            "role": "One invoice document",
            "kind": "section",
            "example": "",
            "examples": [],
            "parent_role": "",
        }
    }
}
_XML = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"'
    ' xmlns:docset="http://dgml.io/test/SyntheticNs">'
    '<docset:Invoice dg:structure="section">text</docset:Invoice>'
    "</dg:chunk>"
)


def test_generation_schema_is_read_back_from_where_it_was_written(
    s3_workspace: Workspace,
) -> None:
    """The defect that motivated this whole package.

    ``docsets/<id>/schema.json`` is written as a *blob* (exact ``Schema.save``
    bytes) and was once read back with ``get_doc``. On local disk both resolve to
    one path, so it worked; here the blob store and the document store are
    different systems, and reading from the wrong one returns nothing —
    ``write_docset_rnc`` would silently produce no schema at all."""
    store = s3_workspace.store
    docset_id = "d100000000dd"
    store.put_doc(layout.Collection.DOCSETS, docset_id, {"id": docset_id, "name": "Invoices"})
    store.put_blob(
        layout.docset_generation_schema_key(docset_id), json.dumps(_SCHEMA).encode("utf-8")
    )
    store.put_blob(layout.dgml_xml_key(docset_id, "f100000000ff", "doc"), _XML.encode("utf-8"))

    full_key = write_docset_rnc(s3_workspace, docset_id)

    assert full_key == layout.docset_full_schema_key(docset_id)
    assert b"Invoice" in store.get_blob(full_key)


def test_write_docset_rnc_returns_none_without_a_schema(s3_workspace: Workspace) -> None:
    s3_workspace.store.put_doc(layout.Collection.DOCSETS, "d1", {"id": "d1", "name": "X"})
    assert write_docset_rnc(s3_workspace, "d1") is None


# ---------------------------------------------------------------- attestation


@needs_gs
def test_attest_export_and_verify(
    s3_workspace: Workspace, make_pdf: Callable[..., Path], tmp_path: Path
) -> None:
    """Proof-of-origin end to end on a remote backend: collect the slots, roll
    them into a Merkle root, export a DGMLX bundle, verify it."""
    file_id = _add_file(s3_workspace, make_pdf, tmp_path, pages=2)

    version = collect_file_version(s3_workspace, file_id)
    assert [a.slot_id for a in version.artifacts] == [
        "source",
        "page_image[1]",
        "page_image[2]",
    ]

    attestation = attest_file(s3_workspace, file_id)
    assert attestation.root

    out_dir = tmp_path / "bundle"
    export_attestation(s3_workspace, file_id, out_dir, None, unpacked=True)
    assert verify_bundle(out_dir).valid is True


# ----------------------------------------------------------------- usage log


def test_usage_log_appends(s3_workspace: Workspace) -> None:
    """``append_doc`` is the one append-only collection."""
    for op in ("classify", "extract_values"):
        record_usage(
            s3_workspace,
            UsageEvent(
                at="2026-01-01T00:00:00Z",
                operation=op,
                model="m",
                cost_usd=0.01,
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
                duration_s=0.5,
                outcome="ok",
            ),
        )
    assert [e["operation"] for e in read_events(s3_workspace)] == ["classify", "extract_values"]
