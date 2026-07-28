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

"""The bundled local-disk :class:`StorageService`.

Maps both APIs onto the **existing** workspace directory layout (see
``docs/storage-layout.md``), so a local workspace on disk is byte-for-byte what
it is today — no migration, and everything that reads the tree directly
(``dgml check``, attestation, DGMLX bundles, external tooling) keeps working.

- **Blob keys are the on-disk relative paths** themselves: a blob is stored at
  ``<root>/<key>`` — ``files/<id>/page_images/page_1.png``, ``files/<id>/<name>``,
  ``files/<id>/page_text/page_1.json`` (bulky per-page word boxes, a blob like the
  page images despite the ``.json`` name), ``docsets/<did>/files/<fid>/<stem>.dgml.xml``,
  ``docsets/<did>/full-schema.rnc``.
- **JSON documents map by ``(collection, id)`` to their real manifest paths**:
  ``files`` → ``files/<id>/file.json``, ``docsets`` → ``docsets/<id>/docset.json``,
  and so on. The document is stored **verbatim** — no ``_id`` is injected, so
  ``file.json`` is exactly the ``FileRecord`` JSON it is today. Two collections are
  special-cased: ``usage`` is the append-only ``usage.jsonl`` (one JSON object per
  line), and ``assignments`` is today's **empty marker directory**
  ``docsets/<did>/files/<fid>/`` — its ``{docset_id, file_id}`` body is reconstructed
  from the path, so no ``assignment.json`` file is written.

Blobs and documents interleave in the same directories, so :meth:`list_blobs`
excludes the recognized document/reserved filenames. Every write is temp-file +
atomic rename.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import CorruptMetadata
from .storage import read_json
from .storage_service import StorageConfig, StorageService

# On-disk directory names (see docs/storage-layout.md).
FILES_DIR = "files"
DOCSETS_DIR = "docsets"
DOCSET_FILES_DIR = "files"  # the per-docset assignment/output dir: docsets/<id>/files/

# Manifest / bootstrap filenames.
FILE_MANIFEST = "file.json"
DOCSET_MANIFEST = "docset.json"
ERRORS_FILE = "errors.json"
GENERATION_SCHEMA_FILE = "schema.json"
WORKSPACE_FILE = "workspace.json"
EXTRACTION_STATS_FILE = "extraction_stats.json"
CONFIG_FILE = "config.json"
USAGE_FILE = "usage.jsonl"


class Collection(StrEnum):
    """The document collections the DGML workspace layout recognizes.

    A ``StrEnum`` so it stays interchangeable with the generic ``collection: str``
    interface — callers may pass ``Collection.FILES`` or ``"files"``, and a
    third-party store can still use any collection name it likes.
    """

    FILES = "files"
    DOCSETS = "docsets"
    WORKSPACE = "workspace"
    SCHEMAS = "schemas"
    ERRORS = "errors"
    ASSIGNMENTS = "assignments"
    EXTRACTION_STATS = "extraction_stats"
    USAGE = "usage"  # append-only; special-cased to usage.jsonl


@dataclass(frozen=True)
class _DocLayout:
    """How a document collection maps onto the on-disk tree.

    ``template`` is a format string over the id parts (e.g. ``"files/{id}/file.json"``,
    ``"docsets/{did}/files/{fid}/extraction_stats.json"``); ``id_parts`` names the
    ``/``-separated segments of ``doc_id`` the template consumes. ``glob`` (derived
    from ``template`` by replacing each placeholder with ``*``) enumerates the
    collection under the workspace root.
    """

    template: str
    id_parts: tuple[str, ...]

    @property
    def glob(self) -> str:
        return re.sub(r"\{[^}]+\}", "*", self.template)


# Directory templates (format strings over id parts), mirroring the ``*_dir``
# helpers on ``Workspace`` in storage.py so the layout lives in one place.
def file_dir_template() -> str:
    """Per-file directory template, e.g. ``files/{id}``."""
    return f"{FILES_DIR}/{{id}}"


def docset_dir_template() -> str:
    """Per-docset directory template, e.g. ``docsets/{id}``."""
    return f"{DOCSETS_DIR}/{{id}}"


def docset_file_dir_template() -> str:
    """Per-(docset, file) directory template, e.g. ``docsets/{did}/files/{fid}``."""
    return f"{DOCSETS_DIR}/{{did}}/{DOCSET_FILES_DIR}/{{fid}}"


# Layout templates for the known collections, composed from the directory templates
# above. ``USAGE`` is absent — it is append-only and handled separately (usage.jsonl).
_DOC_LAYOUTS: dict[str, _DocLayout] = {
    Collection.FILES: _DocLayout(f"{file_dir_template()}/{FILE_MANIFEST}", ("id",)),
    Collection.ERRORS: _DocLayout(f"{file_dir_template()}/{ERRORS_FILE}", ("id",)),
    Collection.DOCSETS: _DocLayout(f"{docset_dir_template()}/{DOCSET_MANIFEST}", ("id",)),
    Collection.SCHEMAS: _DocLayout(f"{docset_dir_template()}/{GENERATION_SCHEMA_FILE}", ("id",)),
    Collection.WORKSPACE: _DocLayout(WORKSPACE_FILE, ()),
    Collection.EXTRACTION_STATS: _DocLayout(
        f"{docset_file_dir_template()}/{EXTRACTION_STATS_FILE}", ("did", "fid")
    ),
}

# Filenames that are documents or bootstrap/config, never returned by ``list_blobs``
# even though they live beside blobs. Derived from the layouts' fixed basenames plus
# the bootstrap files. (page_text is a *blob*, like page images — bulky per-page data
# that is round-tripped, not a queried document — so it is not listed here.)
_NON_BLOB_BASENAMES = frozenset(
    layout.template.rsplit("/", 1)[-1]
    for layout in _DOC_LAYOUTS.values()
    if "{" not in layout.template.rsplit("/", 1)[-1]
) | {CONFIG_FILE, USAGE_FILE}


def _safe_segment(seg: str) -> str:
    """A single path segment, rejecting traversal / separators."""
    if not seg or "/" in seg or seg in (".", "..") or "\\" in seg:
        raise ValueError(f"invalid id/key segment {seg!r}")
    return seg


def _safe_rel(rel: str) -> Path:
    """A workspace-relative POSIX path, rejecting absolute paths and ``..``."""
    if not rel or rel.startswith("/"):
        raise ValueError(f"invalid storage key {rel!r}: must be a non-empty relative path")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"invalid storage key {rel!r}: '..' is not allowed")
    return Path(*parts)


def _split_id(doc_id: str, n: int) -> list[str]:
    """Split a composite document id (e.g. ``"<did>/<fid>"``) into ``n`` safe
    segments."""
    parts = doc_id.split("/")
    if len(parts) != n:
        raise ValueError(f"document id {doc_id!r} must have {n} '/'-separated parts")
    return [_safe_segment(p) for p in parts]


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _matches(doc: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    return all(doc.get(k) == v for k, v in query.items())


class LocalStore(StorageService):
    """Local-disk store over today's workspace layout. Takes no options; its
    location is the workspace root."""

    name = "local"
    config_fields = frozenset()

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        return config

    def __init__(self, config: StorageConfig) -> None:
        self._root = Path(config.root)

    # ---- Blobs (S3-shaped): the key *is* the on-disk relative path ----

    def _blob_path(self, key: str) -> Path:
        return self._root / _safe_rel(key)

    def put_blob(self, key: str, data: bytes) -> None:
        _write_bytes_atomic(self._blob_path(key), data)

    def get_blob(self, key: str) -> bytes:
        path = self._blob_path(key)
        if not path.is_file():
            raise FileNotFoundError(f"no blob at key {key!r}")
        return path.read_bytes()

    def delete_blob(self, key: str) -> None:
        self._blob_path(key).unlink(missing_ok=True)

    def blob_exists(self, key: str) -> bool:
        return self._blob_path(key).is_file()

    def list_blobs(self, prefix: str) -> list[str]:
        root = self._root
        keys = [
            rel
            for path in root.rglob("*")
            if path.is_file() and self._is_blob(rel := path.relative_to(root).as_posix())
        ]
        return sorted(k for k in keys if k.startswith(prefix))

    @staticmethod
    def _is_blob(rel: str) -> bool:
        parts = rel.split("/")
        name = parts[-1]
        if name.endswith(".tmp"):
            return False
        if name in _NON_BLOB_BASENAMES:
            return False
        return True

    def upload_blob(self, key: str, src: Path) -> None:
        dest = self._blob_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dest)

    def download_blob(self, key: str, dest: Path) -> None:
        src = self._blob_path(key)
        if not src.is_file():
            raise FileNotFoundError(f"no blob at key {key!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    # ---- Path bridge — zero-copy: the key already IS an on-disk path ----

    @contextlib.contextmanager
    def materialize(self, key: str) -> Iterator[Path]:
        path = self._blob_path(key)
        if not path.is_file():
            raise FileNotFoundError(f"no blob at key {key!r}")
        yield path

    @contextlib.contextmanager
    def staged_write(self, key_prefix: str) -> Iterator[Path]:
        # The staging dir IS the destination, so the tool renders final bytes in
        # place — no upload step (the base default would copy temp → root).
        dest = self._blob_path(key_prefix.rstrip("/"))
        dest.mkdir(parents=True, exist_ok=True)
        yield dest

    def delete_blobs(self, prefix: str) -> None:
        # Remove only blob files under the prefix (documents that live beside them —
        # file.json, extraction_stats.json, … — are left for delete_doc), then prune
        # any directories emptied as a result so the tree matches a recursive remove.
        base = self._root / _safe_rel(prefix)
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and self._is_blob(path.relative_to(self._root).as_posix()):
                    path.unlink()
        elif base.is_file() and self._is_blob(base.relative_to(self._root).as_posix()):
            base.unlink()
        self._prune_empty_dirs(base)

    def _is_assignment_marker(self, directory: Path) -> bool:
        """Whether ``directory`` is a ``docsets/<did>/files/<fid>`` pair directory —
        an *empty one is itself a live assignment* (see the assignments collection),
        so pruning must never remove it; only ``delete_doc("assignments", …)`` does."""
        try:
            parts = directory.relative_to(self._root).parts
        except ValueError:
            return False
        return len(parts) == 4 and parts[0] == DOCSETS_DIR and parts[2] == DOCSET_FILES_DIR

    def _prune_empty_dirs(self, base: Path) -> None:
        """Remove empty directories in and above ``base`` (bottom-up), stopping at
        the workspace root's top-level directories (``files/``, ``docsets/``) and the
        root itself — so composed blob+document deletes leave no lingering empty dirs
        (matching the historical recursive remove). Assignment marker directories are
        preserved: an empty one is a live assignment, not garbage."""
        if base.is_dir():
            subdirs = sorted(
                (p for p in base.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            )
            for sub in subdirs:
                if self._is_assignment_marker(sub):
                    continue
                with contextlib.suppress(OSError):
                    sub.rmdir()
        directory = base
        while directory != self._root and directory.parent != self._root:
            if self._is_assignment_marker(directory):
                break
            try:
                parent = directory.parent
                directory.rmdir()
            except OSError:
                break  # not empty (or already gone) → stop
            directory = parent

    # ---- JSON documents (Mongo-shaped): mapped to today's manifest paths ----

    def _assignment_dir(self, doc_id: str) -> Path:
        """The per-(docset, file) marker directory for an assignment id ``did/fid``."""
        did, fid = _split_id(doc_id, 2)
        return self._root / DOCSETS_DIR / did / DOCSET_FILES_DIR / fid

    def _doc_path(self, collection: str, doc_id: str) -> Path:
        layout = _DOC_LAYOUTS.get(collection)
        if layout is None:
            # Unknown collection: a generic per-id file, kept out of the blob
            # namespace by its ``.json`` extension under a same-named directory.
            return self._root / _safe_segment(collection) / f"{_safe_segment(doc_id)}.json"
        if not layout.id_parts:
            rel = layout.template
        else:
            segments = _split_id(doc_id, len(layout.id_parts))
            rel = layout.template.format(**dict(zip(layout.id_parts, segments, strict=True)))
        return self._root / _safe_rel(rel)

    def insert_doc(self, collection: str, doc: dict[str, Any]) -> None:
        if collection == Collection.USAGE:
            line = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
            path = self._root / USAGE_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return
        doc_id = doc.get("_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"insert_doc into {collection!r} requires a string '_id'")
        # ``_id`` routes the write; it is not persisted (manifests stay verbatim).
        self.put_doc(collection, doc_id, {k: v for k, v in doc.items() if k != "_id"})

    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        if collection == Collection.USAGE:
            return None  # append-only; read via find_docs, not by id
        if collection == Collection.ASSIGNMENTS:
            did, fid = _split_id(doc_id, 2)
            if not self._assignment_dir(doc_id).is_dir():
                return None
            return {"docset_id": did, "file_id": fid}
        path = self._doc_path(collection, doc_id)
        if not path.is_file():
            return None
        return self._read_doc(path)

    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [doc for doc in self._iter_docs(collection) if _matches(doc, query)]

    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        if collection == Collection.ASSIGNMENTS:
            # An assignment is an empty marker directory (as today); its
            # ``{docset_id, file_id}`` body is reconstructed from the path, so the
            # doc body is not persisted.
            self._assignment_dir(doc_id).mkdir(parents=True, exist_ok=True)
            return
        # Stored verbatim — the manifest keeps its own fields (e.g. ``id``); no
        # ``_id`` is injected, so ``file.json`` is byte-identical to today.
        _write_text_atomic(
            self._doc_path(collection, doc_id),
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        )

    def delete_doc(self, collection: str, doc_id: str) -> None:
        if collection == Collection.USAGE:
            return
        if collection == Collection.ASSIGNMENTS:
            # Matches the historical remove_file: drop the whole pair directory
            # (marker + any generated dgml.xml / extraction_stats inside).
            shutil.rmtree(self._assignment_dir(doc_id), ignore_errors=True)
            return
        self._doc_path(collection, doc_id).unlink(missing_ok=True)

    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        if collection == Collection.USAGE:
            path = self._root / USAGE_FILE
            if not path.is_file():
                return 0
            docs = list(self._iter_docs(collection))
            kept = [doc for doc in docs if not _matches(doc, query)]
            _write_text_atomic(
                path, "".join(json.dumps(d, separators=(",", ":")) + "\n" for d in kept)
            )
            return len(docs) - len(kept)
        if collection == Collection.ASSIGNMENTS:
            matched = self.find_docs(collection, query)
            for doc in matched:
                self.delete_doc(collection, f"{doc['docset_id']}/{doc['file_id']}")
            return len(matched)
        removed = 0
        for path in self._doc_paths(collection):
            doc = self._read_doc(path)
            if _matches(doc, query):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    # ---- document enumeration ----

    def _doc_paths(self, collection: str) -> list[Path]:
        layout = _DOC_LAYOUTS.get(collection)
        pattern = layout.glob if layout is not None else f"{collection}/*.json"
        return sorted(p for p in self._root.glob(pattern) if p.is_file())

    def _iter_docs(self, collection: str) -> Iterator[dict[str, Any]]:
        if collection == Collection.USAGE:
            path = self._root / USAGE_FILE
            if not path.is_file():
                return
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a corrupt tail line from a crashed append
                if isinstance(obj, dict):
                    yield obj
            return
        if collection == Collection.ASSIGNMENTS:
            # Enumerate the per-(docset, file) marker directories, reconstructing
            # each assignment's body from the path.
            pattern = f"{DOCSETS_DIR}/*/{DOCSET_FILES_DIR}/*"
            for path in sorted(self._root.glob(pattern)):
                if path.is_dir():
                    yield {"docset_id": path.parent.parent.name, "file_id": path.name}
            return
        for path in self._doc_paths(collection):
            try:
                yield self._read_doc(path)
            except CorruptMetadata:
                # A corrupt manifest reads as absent for enumeration (matches the
                # historical list_all behavior of skipping unparseable docsets).
                continue

    @staticmethod
    def _read_doc(path: Path) -> dict[str, Any]:
        # ``read_json`` gives duplicate-key rejection and raises CorruptMetadata on
        # bad JSON — the same contract the manifest readers relied on.
        obj = read_json(path)
        if not isinstance(obj, dict):
            raise ValueError(f"document {path} is not a JSON object")
        return obj
