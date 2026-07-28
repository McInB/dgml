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

"""Tests for the pluggable StorageService: LocalStore, resolver, config, fingerprint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dgml_core import (
    DEFAULT_STORAGE_PROVIDER,
    LocalStore,
    StorageConfig,
    StorageService,
    Workspace,
    load_storage_config,
    make_store,
    storage_fingerprint,
)
from dgml_core.errors import StorageConfigInvalid, StorageProviderUnresolvable


def local_store(root: Path) -> LocalStore:
    cfg = StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=root)
    return LocalStore(LocalStore.parse_config(cfg))


# --------------------------------------------------------------------------- blobs


def test_blob_maps_to_real_on_disk_path(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    key = "files/abc/page_images/page_1.png"
    assert store.blob_exists(key) is False
    store.put_blob(key, b"\x89PNG-data")
    assert store.blob_exists(key) is True
    assert store.get_blob(key) == b"\x89PNG-data"
    # the blob is at the exact legacy path — no sandbox prefix
    assert (
        tmp_path / "files" / "abc" / "page_images" / "page_1.png"
    ).read_bytes() == b"\x89PNG-data"
    # put overwrites (S3 semantics = update)
    store.put_blob(key, b"replaced")
    assert store.get_blob(key) == b"replaced"


def test_blob_missing_get_raises(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get_blob("files/x/page_images/page_1.png")


def test_blob_delete_is_idempotent(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/a/report.pdf", b"1")
    store.delete_blob("files/a/report.pdf")
    store.delete_blob("files/a/report.pdf")  # no error on missing
    assert store.blob_exists("files/a/report.pdf") is False


def test_list_blobs_excludes_documents(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/f1/report.pdf", b"pdf")
    store.put_blob("files/f1/page_images/page_1.png", b"png")
    store.put_blob("files/f1/page_text/page_1.json", b'{"words": []}')  # a blob, despite .json
    store.put_doc("files", "f1", {"id": "f1"})  # -> files/f1/file.json (a document)
    # blobs (incl. page_text) are listed; only the file.json manifest is excluded
    assert store.list_blobs("files/f1/") == [
        "files/f1/page_images/page_1.png",
        "files/f1/page_text/page_1.json",
        "files/f1/report.pdf",
    ]


def test_delete_blobs_is_blob_only_and_prunes(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/f1/report.pdf", b"pdf")
    store.put_blob("files/f1/page_images/page_1.png", b"png")
    store.put_doc("files", "f1", {"id": "f1"})
    store.put_blob("files/f2/report.pdf", b"other")

    store.delete_blobs("files/f1/")
    # blobs under the prefix are gone; the document beside them is untouched
    assert store.list_blobs("files/f1/") == []
    assert store.get_doc("files", "f1") == {"id": "f1"}
    # the emptied blob subdir is pruned; the file dir stays (still holds file.json)
    assert not (tmp_path / "files" / "f1" / "page_images").exists()
    assert (tmp_path / "files" / "f1").is_dir()
    # a sibling under the same parent is untouched; a missing prefix is a no-op
    assert store.get_blob("files/f2/report.pdf") == b"other"
    store.delete_blobs("files/nope/")

    # once the document is gone too, delete_blobs prunes the now-empty file dir
    store.delete_doc("files", "f1")
    store.delete_blobs("files/f1/")
    assert not (tmp_path / "files" / "f1").exists()


def test_delete_blobs_preserves_assignment_markers(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    # an assignment is an empty marker dir; a generated dgml.xml blob sits beside it
    store.insert_doc("assignments", {"_id": "d1/f1", "docset_id": "d1", "file_id": "f1"})
    store.put_blob("docsets/d1/files/f1/report.dgml.xml", b"<x/>")
    store.delete_blobs("docsets/d1/files/f1/")
    # the blob is gone, but pruning must not remove the marker → assignment survives
    assert not store.blob_exists("docsets/d1/files/f1/report.dgml.xml")
    assert store.get_doc("assignments", "d1/f1") == {"docset_id": "d1", "file_id": "f1"}
    assert (tmp_path / "docsets" / "d1" / "files" / "f1").is_dir()


def test_upload_download_blob(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    store.upload_blob("files/d/e.bin", src)
    assert store.get_blob("files/d/e.bin") == b"payload"
    dest = tmp_path / "out" / "dl.bin"
    store.download_blob("files/d/e.bin", dest)
    assert dest.read_bytes() == b"payload"


def test_blob_key_traversal_rejected(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    for bad in ("/abs/key", "../escape", "a/../../b"):
        with pytest.raises(ValueError):
            store.put_blob(bad, b"x")


# ----------------------------------------------------------------------- documents


def test_manifest_stored_verbatim_at_legacy_path(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    record = {"id": "f1", "sha256": "aa", "page_count": 2}
    store.put_doc("files", "f1", record)
    # byte-for-byte the FileRecord JSON — no injected ``_id``, at the real path
    on_disk = json.loads((tmp_path / "files" / "f1" / "file.json").read_text(encoding="utf-8"))
    assert on_disk == record
    assert store.get_doc("files", "f1") == record


def test_doc_put_get_update(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    assert store.get_doc("files", "f1") is None
    store.put_doc("files", "f1", {"id": "f1", "sha256": "aa"})
    assert store.get_doc("files", "f1") == {"id": "f1", "sha256": "aa"}
    # put replaces (update)
    store.put_doc("files", "f1", {"id": "f1", "sha256": "bb"})
    assert store.get_doc("files", "f1") == {"id": "f1", "sha256": "bb"}


def test_docset_and_workspace_collections(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_doc("docsets", "d1", {"id": "d1", "name": "Contracts"})
    assert (tmp_path / "docsets" / "d1" / "docset.json").is_file()
    store.put_doc("workspace", "workspace", {"name": "W", "organization": "Acme"})
    assert json.loads((tmp_path / "workspace.json").read_text())["organization"] == "Acme"


def test_assignments_as_empty_marker_dirs(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.insert_doc("assignments", {"_id": "d1/f1", "docset_id": "d1", "file_id": "f1"})
    store.insert_doc("assignments", {"_id": "d1/f2", "docset_id": "d1", "file_id": "f2"})
    store.insert_doc("assignments", {"_id": "d2/f1", "docset_id": "d2", "file_id": "f1"})
    # on disk it's today's empty marker directory — no assignment.json file
    pair = tmp_path / "docsets" / "d1" / "files" / "f1"
    assert pair.is_dir()
    assert not (pair / "assignment.json").exists()
    assert list(pair.iterdir()) == []
    # the body is reconstructed from the path
    assert store.get_doc("assignments", "d1/f1") == {"docset_id": "d1", "file_id": "f1"}
    assert store.get_doc("assignments", "d1/nope") is None
    # both relationship directions are queryable
    assert sorted(d["file_id"] for d in store.find_docs("assignments", {"docset_id": "d1"})) == [
        "f1",
        "f2",
    ]
    assert sorted(d["docset_id"] for d in store.find_docs("assignments", {"file_id": "f1"})) == [
        "d1",
        "d2",
    ]
    assert len(store.find_docs("assignments", {})) == 3
    # delete removes the whole pair dir (matches the historical remove_file)
    store.delete_doc("assignments", "d1/f1")
    assert not pair.exists()
    assert len(store.find_docs("assignments", {})) == 2


def test_insert_doc_requires_id(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    with pytest.raises(ValueError):
        store.insert_doc("files", {"sha256": "aa"})  # no _id to route the write


def test_delete_doc_and_delete_docs(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_doc("files", "f1", {"id": "f1"})
    store.put_doc("files", "f2", {"id": "f2"})
    store.delete_doc("files", "f1")
    assert store.get_doc("files", "f1") is None
    store.delete_doc("files", "missing")  # no error

    for did, fid in [("d1", "f1"), ("d1", "f2"), ("d2", "f1")]:
        store.insert_doc("assignments", {"_id": f"{did}/{fid}", "docset_id": did, "file_id": fid})
    removed = store.delete_docs("assignments", {"docset_id": "d1"})
    assert removed == 2
    assert len(store.find_docs("assignments", {})) == 1


def test_usage_is_append_only(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.insert_doc("usage", {"op": "transcribe", "cost_usd": 0.01})
    store.insert_doc("usage", {"op": "label", "cost_usd": 0.02})
    events = store.find_docs("usage", {})
    assert [e["op"] for e in events] == ["transcribe", "label"]
    assert "_id" not in events[0]  # stored verbatim as JSONL
    assert (tmp_path / "usage.jsonl").exists()
    assert store.find_docs("usage", {"op": "label"}) == [{"op": "label", "cost_usd": 0.02}]
    assert store.delete_docs("usage", {"op": "transcribe"}) == 1
    assert [e["op"] for e in store.find_docs("usage", {})] == ["label"]


def test_usage_tolerates_corrupt_tail(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.insert_doc("usage", {"op": "ok"})
    with (tmp_path / "usage.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"op": "truncated"')  # crashed mid-append, no newline / close brace
    assert [e["op"] for e in store.find_docs("usage", {})] == ["ok"]


# ------------------------------------------------------------------ resolver/config


def test_make_store_default_is_local(tmp_path: Path) -> None:
    store = make_store(StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=tmp_path))
    assert isinstance(store, LocalStore)
    assert isinstance(store, StorageService)


def test_make_store_bad_provider() -> None:
    for bad in ["noColon", "no.module.here.at.all:Class", "json:Nonexistent"]:
        with pytest.raises(StorageProviderUnresolvable):
            make_store(StorageConfig(provider=bad, root=Path(".")))


def test_make_store_not_a_storage_subclass() -> None:
    # importable + resolvable, but not a StorageService
    with pytest.raises(StorageProviderUnresolvable):
        make_store(StorageConfig(provider="json:JSONDecoder", root=Path(".")))


def test_local_store_rejects_unknown_options(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigInvalid):
        LocalStore.parse_config(
            StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=tmp_path, options={"bucket": "x"})
        )


def test_load_storage_config_defaults_to_local(tmp_path: Path) -> None:
    ws = Workspace.resolve(tmp_path)
    cfg = load_storage_config(ws)
    assert cfg.provider == DEFAULT_STORAGE_PROVIDER
    assert cfg.root == ws.root


def test_load_storage_config_reads_section(tmp_path: Path) -> None:
    ws = Workspace.resolve(tmp_path)
    ws.config_path.write_text(
        '{"storage": {"provider": "my_pkg.store:MyStore", "bucket": "b1"}}', encoding="utf-8"
    )
    cfg = load_storage_config(ws)
    assert cfg.provider == "my_pkg.store:MyStore"
    assert cfg.options == {"bucket": "b1"}


def test_load_storage_config_invalid_provider(tmp_path: Path) -> None:
    ws = Workspace.resolve(tmp_path)
    ws.config_path.write_text('{"storage": {"provider": ""}}', encoding="utf-8")
    with pytest.raises(StorageConfigInvalid):
        load_storage_config(ws)


# --------------------------------------------------------------------- fingerprint


def test_fingerprint_stable_and_location_sensitive() -> None:
    root = Path("/tmp/ws")
    a = StorageConfig(provider="p:C", root=root, options={"bucket": "b", "prefix": "x"})
    a2 = StorageConfig(provider="p:C", root=Path("/other"), options={"prefix": "x", "bucket": "b"})
    b = StorageConfig(provider="p:C", root=root, options={"bucket": "OTHER", "prefix": "x"})
    # stable across option order and independent of the local root
    assert storage_fingerprint(a) == storage_fingerprint(a2)
    # trips when the location changes
    assert storage_fingerprint(a) != storage_fingerprint(b)


def test_fingerprint_ignores_credential_rotation() -> None:
    root = Path("/tmp/ws")
    a = StorageConfig(provider="p:C", root=root, options={"bucket": "b", "api_key": "OLD"})
    b = StorageConfig(provider="p:C", root=root, options={"bucket": "b", "api_key": "NEW"})
    assert storage_fingerprint(a) == storage_fingerprint(b)


def test_third_party_plugin_resolves_by_dotted_path() -> None:
    # dgml_core.storage_local:LocalStore is resolved exactly like a third party's
    # own dotted path — proving the plug-in mechanism end to end.
    cfg = StorageConfig(provider="dgml_core.storage_local:LocalStore", root=Path("."))
    assert isinstance(make_store(cfg), LocalStore)
