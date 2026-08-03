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

"""Pluggable workspace storage — a blob + JSON ``StorageService``.

The service handles two kinds of data, each with a small, familiar API:

- **Blobs** (opaque bytes — page images, PDFs, XML, schema files): modeled on
  the S3 object API (``put_blob`` / ``get_blob`` / ``list_blobs`` / …).
- **JSON documents** (manifests, page text, assignments, usage): modeled on the
  MongoDB collection API (``insert_doc`` / ``get_doc`` / ``find_docs`` / …).

Both kinds support create / read / update / delete.

A ``provider`` is a dotted ``"module.path:ClassName"`` that :func:`make_store`
imports at runtime and checks is a :class:`StorageService` subclass — exactly
like a :class:`dgml_core.conversion.DocConverter`. This module ships only the
abstraction and the resolver; the one bundled implementation,
:class:`dgml_core.storage_local.LocalStore`, is referenced by its dotted path
like any third party's own.

Writing your own store
----------------------

1. ``pip install dgml`` (the wheel — no repo clone).
2. Subclass :class:`StorageService`, implementing :meth:`~StorageService.parse_config`
   (call :meth:`~StorageService._check_no_extra_fields` first), ``__init__`` (lazy
   SDK import — raise an actionable error if a dependency is missing), and the blob
   and document methods.
3. Make the class importable by the interpreter running dgml.
4. Set ``storage.provider`` to ``"your_pkg.mod:YourStore"`` in ``config.json``.

The path bridge (:meth:`~StorageService.materialize` and friends) and
:meth:`~StorageService.sha256_blob` are concrete — you get working versions from
the abstract methods above and only override them if your backend can do better.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .errors import StorageConfigInvalid, StorageProviderUnresolvable
from .hashing import sha256_file
from .storage import Workspace, read_config

# The bundled default: local disk. Used when ``config.json`` has no ``storage``
# section. Resolved through the same path as any third-party provider.
DEFAULT_STORAGE_PROVIDER = "dgml_core.storage_local:LocalStore"

# Option keys never folded into the store fingerprint — rotating a credential
# must not read as "the store moved".
_SECRET_HINTS = ("key", "secret", "token", "password", "credential")


@dataclass(frozen=True)
class StorageConfig:
    """A resolved ``storage`` config section.

    ``provider`` is the dotted path identifying the store class. ``options`` holds
    the section's remaining (non-``provider``) fields verbatim — a provider's own
    settings (``bucket``, ``endpoint_url``, …). ``root`` is the local workspace
    root, always available as bootstrap (``config.json`` names the store, so it
    cannot live inside it); a ``LocalStore`` writes under it, and a remote store
    may use it for temp staging.
    """

    provider: str
    root: Path
    options: Mapping[str, Any] = field(default_factory=dict)


class StorageService(ABC):
    """Common interface for pluggable workspace storage backends.

    Subclasses declare ``config_fields`` — the JSON keys they accept under
    ``storage.*`` besides the universal ``provider`` — and are rejected for any
    other key by :meth:`_check_no_extra_fields` (catches typos and stale fields).
    """

    name: ClassVar[str]
    config_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def _check_no_extra_fields(cls, options: Mapping[str, Any]) -> None:
        """Raise :class:`StorageConfigInvalid` for any option key not in
        ``cls.config_fields``."""
        unknown = set(options) - cls.config_fields
        if unknown:
            raise StorageConfigInvalid(
                f"unknown fields in 'storage' for provider {cls.name!r}: "
                f"{sorted(unknown)}. Allowed: {sorted(cls.config_fields)}"
            )

    @classmethod
    @abstractmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        """Validate the provider's option fields and return the (possibly
        normalized) config. Call :meth:`_check_no_extra_fields` first; raise
        :class:`StorageConfigInvalid` for missing or malformed fields."""

    @abstractmethod
    def __init__(self, config: StorageConfig) -> None:
        """Set the store up from ``config``. Lazy-import any SDK here and raise an
        actionable :class:`dgml_core.errors.DgmlError` if it is missing."""

    # ---- Blobs — modeled on the S3 object API (key -> bytes) ----

    @abstractmethod
    def put_blob(self, key: str, data: bytes) -> None:
        """Create or overwrite the blob at ``key`` (S3 ``put_object``)."""

    @abstractmethod
    def get_blob(self, key: str) -> bytes:
        """Return the blob at ``key``. Raise :class:`FileNotFoundError` if absent.

        Returns the whole blob in memory — fine for DGML's artifact sizes (PDFs,
        page images, schemas, one dgml.xml), which is the working assumption
        throughout. Use this only when the bytes themselves are needed (parsing
        XML, base64-encoding an image, ``json.loads``). A caller that only needs
        a digest should use :meth:`sha256_blob`; one that needs a real path
        should use :meth:`download_blob` / :meth:`materialize`. Both avoid
        holding the blob whole."""

    @abstractmethod
    def delete_blob(self, key: str) -> None:
        """Delete the blob at ``key``. A missing key is a no-op (idempotent)."""

    @abstractmethod
    def blob_exists(self, key: str) -> bool:
        """Whether a blob exists at ``key`` (S3 ``head_object``)."""

    @abstractmethod
    def list_blobs(self, prefix: str) -> list[str]:
        """All blob keys under ``prefix`` (S3 ``list_objects_v2``), sorted."""

    @abstractmethod
    def upload_blob(self, key: str, src: Path) -> None:
        """Store the file at ``src`` as the blob ``key`` (S3 ``upload_file``)."""

    @abstractmethod
    def download_blob(self, key: str, dest: Path) -> None:
        """Write the blob ``key`` to the local path ``dest`` (S3 ``download_file``)."""

    # ---- Path bridge — for tools that demand a real filesystem path ----
    #
    # Some pipeline steps speak *paths*, not bytes: ghostscript renders page
    # images to ``-sOutputFile=<dir>/page_%d.png``, pdfminer / the PDF converter
    # read a PDF path, ``lxml.etree.parse`` wants a path. These concrete helpers
    # bridge that gap on top of the blob primitives, so every store gets them for
    # free; ``LocalStore`` overrides each one for a zero-copy passthrough (the key
    # already *is* an on-disk path), keeping local I/O byte-for-byte identical to
    # the pre-store code.
    #
    # A remote store overriding these should stage under ``StorageConfig.root``
    # rather than the default ``tempfile`` location: ``TMPDIR`` is a RAM-backed
    # tmpfs on many container images, which would silently turn a bounded-memory
    # read back into a whole-blob allocation plus a copy.

    @contextmanager
    def materialize(self, key: str) -> Iterator[Path]:
        """Yield a real local path holding the blob at ``key`` for a
        path-only reader (ghostscript, pdfminer, ``lxml.etree.parse``).

        Default: download to a temp file, cleaned up on exit. ``LocalStore``
        yields the real file with no copy. Raises :class:`FileNotFoundError`
        if the blob is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / Path(key).name
            self.download_blob(key, dest)
            yield dest

    @contextmanager
    def staged_write(self, key_prefix: str) -> Iterator[Path]:
        """Yield an empty local directory for a tool that emits a *batch* of
        files by path (ghostscript rendering a file's page images).

        **The prefix is replaced, not added to.** On clean exit the blobs under
        ``key_prefix`` are *exactly* the files written into the yielded
        directory: everything written is stored (preserving relative paths) and
        any pre-existing blob under the prefix that was not rewritten is
        deleted. If the body raises, nothing is persisted and the prefix is left
        as it was.

        Replacement is part of the contract rather than an implementation
        detail because the callers regenerate a whole set at once — re-render a
        document whose page count dropped from 10 to 5 and the stale
        ``page_6..10`` must not survive. They would otherwise be hashed into the
        file's attestation, so a purely additive implementation makes the Merkle
        root depend on which backend the workspace happens to live on.

        A store overriding this must keep both halves of the contract: an
        **empty** directory on entry, and an exact replacement on exit."""
        prefix = key_prefix.rstrip("/")
        stale = set(self.list_blobs(prefix + "/"))
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            yield staging
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(staging).as_posix()
                    key = f"{prefix}/{rel}"
                    self.upload_blob(key, path)
                    stale.discard(key)
            for key in sorted(stale):
                self.delete_blob(key)

    @contextmanager
    def materialize_dir(self, prefix: str) -> Iterator[Path]:
        """Yield a local directory holding every blob under ``prefix`` (each at
        its path relative to ``prefix``), for a tool that *scans a directory* of
        files (OCR reading a file's rendered page images).

        Default: download the matching blobs into a temp dir, cleaned up on
        exit. ``LocalStore`` yields the real directory with no copy. The
        directory may be empty/absent if nothing matches — the caller handles
        that (OCR raises its own \"no page images\" error)."""
        base = prefix.rstrip("/") + "/"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for key in self.list_blobs(base):
                self.download_blob(key, out / key[len(base) :])
            yield out

    @contextmanager
    def working_dir(self, prefix: str) -> Iterator[Path]:
        """Yield a local, read-write working directory synced with ``prefix``:
        download its blobs in on entry, upload the directory's files back out on
        exit. For a read-modify-write working area the pipeline reloads across
        runs (the generation ``cache/``).

        As with :meth:`staged_write`, the sync back is a **replacement**: a blob
        the body deleted locally is deleted from the store, not silently
        resurrected on the next run.

        The yielded directory is named after the last segment of ``prefix`` and
        lives inside a fresh temp dir, so its *parent* is a stable per-call
        scratch location — a sibling artifact written next to it (generation's
        ``schema.json``) has somewhere to go, and is deliberately *not* synced.
        Default: temp dir, downloaded in and uploaded out. ``LocalStore`` yields
        the real directory (no copy, no sync — writes and deletes already land
        in the store).

        Unlike :meth:`staged_write`, a crash does not roll back identically
        across stores: the default persists nothing (the upload runs after the
        ``yield``, not in a ``finally``), while ``LocalStore`` has already
        written in place. That is tolerated because the only caller is a
        regenerable cache — do not use this for artifacts that must not be
        half-written."""
        base = prefix.rstrip("/") + "/"
        segment = prefix.rstrip("/").rsplit("/", 1)[-1] or "data"
        stale = set(self.list_blobs(base))
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / segment
            work.mkdir(parents=True, exist_ok=True)
            for key in stale:
                self.download_blob(key, work / key[len(base) :])
            yield work
            for path in sorted(work.rglob("*")):
                if path.is_file():
                    key = base + path.relative_to(work).as_posix()
                    self.upload_blob(key, path)
                    stale.discard(key)
            for key in sorted(stale):
                self.delete_blob(key)

    # ---- Derived reads — composed from the primitives above ----

    def sha256_blob(self, key: str) -> str:
        """Return the lowercase hex SHA-256 digest of the blob at ``key``.

        The digest of the blob's **exact stored bytes** — the same value as
        ``hashlib.sha256(self.get_blob(key)).hexdigest()``, computed without ever
        holding the whole blob in memory. This is what attestation leaves are
        built from, so it is part of DGML's on-chain contract: an override MUST
        return the plain SHA-256 of the full byte sequence and never a derived
        checksum (S3's multipart ETag and composite ``ChecksumSHA256`` are
        checksums-of-checksums and are **not** this value).

        Default: :meth:`materialize` plus the chunked
        :func:`dgml_core.hashing.sha256_file`. On ``LocalStore`` that is
        zero-copy — the key already *is* an on-disk path, so the real file is
        read in fixed-size chunks with no temp copy and no whole-blob
        allocation. On a remote store it is a managed (ranged, retryable)
        download to a temp file rather than one long-lived response body, which
        is what makes hashing a large artifact reliable there. Raises
        :class:`FileNotFoundError` if the blob is absent."""
        with self.materialize(key) as path:
            return sha256_file(path)

    # ---- JSON documents — modeled on the MongoDB collection API ----

    @abstractmethod
    def insert_doc(self, collection: str, doc: dict[str, Any]) -> None:
        """Add a document to ``collection`` (Mongo ``insert_one``). Except in
        append-only collections (e.g. ``usage``), ``doc`` carries an ``_id``."""

    @abstractmethod
    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        """Return the document with ``_id == doc_id`` (Mongo ``find_one``), or None."""

    @abstractmethod
    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        """All documents in ``collection`` matching every field in ``query``
        (Mongo ``find``). An empty ``query`` returns the whole collection."""

    @abstractmethod
    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        """Insert or replace the document with ``_id == doc_id`` (Mongo
        ``replace_one(upsert=True)``) — this is the update path."""

    @abstractmethod
    def delete_doc(self, collection: str, doc_id: str) -> None:
        """Delete the document with ``_id == doc_id``. Missing is a no-op."""

    @abstractmethod
    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        """Delete every document in ``collection`` matching ``query`` (Mongo
        ``delete_many``). Returns the number deleted."""

    @abstractmethod
    def delete_blobs(self, prefix: str) -> None:
        """Delete every **blob** whose key is under ``prefix`` (an object store: list
        + batch-delete; ``LocalStore``: remove the blob files and prune now-empty
        directories). Documents are left untouched — a cascade delete composes this
        with ``delete_doc`` / ``delete_docs`` in the caller, so each store only ever
        does operations native to it (no store needs the blob/document layout). A
        prefix that matches nothing is a no-op."""


def _resolve_store_class(provider: str) -> type[StorageService]:
    """Import and return the :class:`StorageService` subclass named by ``provider``.

    ``provider`` must be a dotted path ``"module.path:ClassName"``. Raises
    :class:`StorageProviderUnresolvable` if the string is malformed, the module or
    attribute can't be imported, or the target is not a ``StorageService`` subclass.
    """
    if ":" not in provider:
        raise StorageProviderUnresolvable(
            f"storage provider must be a dotted path 'module.path:ClassName' "
            f"(got {provider!r}); the bundled default is {DEFAULT_STORAGE_PROVIDER!r}"
        )
    module_path, _, class_name = provider.partition(":")
    if not module_path or not class_name:
        raise StorageProviderUnresolvable(
            f"storage provider {provider!r} must have the form 'module.path:ClassName'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise StorageProviderUnresolvable(
            f"could not import storage module {module_path!r} for provider {provider!r}: "
            f"{exc}. Is the package installed in this environment?"
        ) from exc
    try:
        obj = getattr(module, class_name)
    except AttributeError as exc:
        raise StorageProviderUnresolvable(
            f"module {module_path!r} has no attribute {class_name!r} (provider {provider!r})"
        ) from exc
    if not (isinstance(obj, type) and issubclass(obj, StorageService)):
        raise StorageProviderUnresolvable(
            f"provider {provider!r} resolved to {obj!r}, which is not a StorageService subclass"
        )
    return obj


def make_store(config: StorageConfig) -> StorageService:
    """Instantiate the store named by ``config``.

    Resolves ``config.provider`` to its class (imported here, not at config-load
    time), runs the provider's :meth:`StorageService.parse_config` to validate its
    fields, then constructs it — where the provider's lazy SDK import happens.
    """
    cls = _resolve_store_class(config.provider)
    return cls(cls.parse_config(config))


def load_storage_config(workspace: Workspace) -> StorageConfig:
    """Read and validate the ``storage`` section of ``<workspace>/config.json``.

    A missing config file or missing ``storage`` section yields the bundled
    default (:data:`DEFAULT_STORAGE_PROVIDER`, local disk). Validates only the
    generic shape — a non-empty string ``provider`` — deferring provider-specific
    field validation to :meth:`StorageService.parse_config` in :func:`make_store`.
    """
    root = workspace.root
    if not workspace.config_path.exists():
        return StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=root)
    try:
        data = read_config(workspace.config_path)
    except Exception as exc:  # CorruptMetadata and friends
        raise StorageConfigInvalid(f"{workspace.config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise StorageConfigInvalid(f"{workspace.config_path} must contain a JSON object")
    section = data.get("storage")
    if section is None:
        return StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=root)
    if not isinstance(section, dict):
        raise StorageConfigInvalid("'storage' must be a JSON object")
    provider = section.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise StorageConfigInvalid("'storage.provider' must be a non-empty string")
    options = {k: v for k, v in section.items() if k != "provider"}
    return StorageConfig(provider=provider, root=root, options=options)


def storage_fingerprint(config: StorageConfig) -> str:
    """A stable, credential-free content hash of the store identity.

    Covers the ``provider`` and its non-secret options (bucket, prefix, endpoint,
    …) so that switching the store trips the guard while rotating a credential
    does not. Sealed at ``workspace create``; recomputed and compared on open (a
    mismatch is :class:`~dgml_core.errors.StorageBackendMismatch`).
    """
    identity = {
        "provider": config.provider,
        "options": {
            k: v
            for k, v in sorted(config.options.items())
            if not any(hint in k.lower() for hint in _SECRET_HINTS)
        },
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()
