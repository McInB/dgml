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
  MongoDB collection API (``put_doc`` / ``get_doc`` / ``find_docs`` / …).

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

from . import layout
from .config import load_merged_config
from .errors import StorageConfigInvalid, StorageProviderUnresolvable
from .hashing import sha256_file
from .models_config import ConfigSection
from .storage import Workspace

# The bundled default: local disk. Used when ``config.json`` has no ``storage``
# section. Resolved through the same path as any third-party provider.
DEFAULT_STORAGE_PROVIDER = "dgml_core.storage_local:LocalStore"

# The storage-service name a workspace uses when none is chosen at create time,
# and the name a bare (unnamed) ``[storage]`` table resolves as.
DEFAULT_STORAGE_SERVICE = "default"

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
                    key = layout.pair_id(prefix, rel)
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
    def append_doc(self, collection: str, doc: dict[str, Any]) -> None:
        """Append ``doc`` to an **append-only** ``collection`` (the usage log).

        Such a document has no id: it is never fetched or replaced individually,
        only enumerated with :meth:`find_docs`. Which collections are append-only
        is the store's own business — ``LocalStore`` backs ``usage`` with
        ``usage.jsonl`` and rejects anything else.

        Deliberately *not* a Mongo-style ``insert_one``: an insert that fails on
        a duplicate id would be a create-if-absent primitive, and nothing in DGML
        needs one (creates go through :meth:`put_doc`, which is idempotent by
        design). Adding it later is easy; shipping a method whose documented
        semantics no implementation honours is not."""

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
        prefix that matches nothing is a no-op.

        Callers must run this **last** in a cascade. That is the contract
        :class:`dgml_core.workspace_ops.WorkspaceOps` implements — *the
        authoritative record dies first*, so an interrupted cascade leaves
        orphaned bytes (recoverable) rather than a record pointing at bytes that
        are gone (indistinguishable from a valid entity). It also happens to be
        what lets ``LocalStore`` prune the emptied container, which it can only
        do once the documents beside those blobs are gone."""


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


def _config_from(section: Mapping[str, Any], root: Path) -> StorageConfig:
    """Build a :class:`StorageConfig` from one service table (``provider`` + the
    rest as ``options``). Raises :class:`StorageConfigInvalid` for a bad shape."""
    provider = section.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise StorageConfigInvalid("'storage.provider' must be a non-empty string")
    options = {k: v for k, v in section.items() if k != "provider"}
    return StorageConfig(provider=provider, root=root, options=options)


def load_storage_config(
    workspace: Workspace, service: str = DEFAULT_STORAGE_SERVICE
) -> StorageConfig:
    """Resolve one **named storage-service template** from the workspace config.

    ``config.toml`` may define several services as ``[storage.<name>]`` subtables;
    ``service`` selects one. Two forms are accepted for back-compat:

    - **Flat** — a bare ``[storage]`` table with a top-level ``provider`` string is
      the single ``"default"`` service (the pre-named-services shape). Asking for
      any other name then raises.
    - **Named** — ``[storage.<name>]`` subtables. ``service`` selects
      ``[storage.<name>]``; an absent ``"default"`` falls back to the bundled
      local-disk store (so an ordinary workspace still needs zero config), while an
      absent *named* service raises.

    Validates only the *generic shape* — ``provider`` is a non-empty string;
    provider resolution and field validation happen lazily in :func:`make_store`,
    so loading the config never imports a backend SDK.

    Raises :class:`StorageConfigInvalid` for a malformed shape or an unknown named
    service.
    """
    root = workspace.root
    section = load_merged_config(workspace).get(ConfigSection.STORAGE) or {}
    if not isinstance(section, dict):
        raise StorageConfigInvalid("'storage' must be a table")
    # Flat form: a top-level ``provider`` string means the whole table is one
    # unnamed store — the "default" service. (``provider`` is reserved at the top
    # of ``[storage]``; a named service is always a subtable.)
    if isinstance(section.get("provider"), str):
        if service != DEFAULT_STORAGE_SERVICE:
            raise StorageConfigInvalid(
                f"no storage service {service!r}: config has a single [storage] table"
            )
        return _config_from(section, root)
    # Named form.
    sub = section.get(service)
    if sub is None:
        if service == DEFAULT_STORAGE_SERVICE:
            # zero-config default: an ordinary workspace runs on local disk.
            return StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=root)
        raise StorageConfigInvalid(f"no [storage.{service}] configured")
    if not isinstance(sub, dict):
        raise StorageConfigInvalid(f"[storage.{service}] must be a table")
    return _config_from(sub, root)


def _identity_hash(provider: str, options: Mapping[str, Any]) -> str:
    """The canonical credential-free store-identity hash — the one hashing scheme
    shared by :func:`storage_fingerprint` and :func:`fingerprint_of_snapshot`."""
    identity = {
        "provider": provider,
        "options": {
            k: v
            for k, v in sorted(options.items())
            if not any(hint in k.lower() for hint in _SECRET_HINTS)
        },
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def storage_fingerprint(config: StorageConfig) -> str:
    """A stable, credential-free content hash of the store identity.

    Covers the ``provider`` and its non-secret options (bucket, prefix, endpoint,
    …) so that switching the store trips the guard while rotating a credential
    does not. Sealed at ``workspace create`` into the registry entry's snapshot;
    recomputed from the entry and compared on open (a mismatch is
    :class:`~dgml_core.errors.StorageBackendMismatch`)."""
    return _identity_hash(config.provider, config.options)


def storage_snapshot(config: StorageConfig) -> dict[str, Any]:
    """The **non-secret** store identity as a flat dict — ``{"provider": …, <opt>:
    …}`` — for persisting into the registry entry. Secret-hinted options are
    dropped, so credentials never reach the plaintext registry. The inverse pair of
    :func:`fingerprint_of_snapshot`."""
    snapshot: dict[str, Any] = {"provider": config.provider}
    snapshot.update(
        (k, v)
        for k, v in config.options.items()
        if not any(hint in k.lower() for hint in _SECRET_HINTS)
    )
    return snapshot


def fingerprint_of_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Recompute the identity hash of a persisted :func:`storage_snapshot`.

    Equal to ``storage_fingerprint`` of the config the snapshot was taken from, so
    the open-time integrity check ``fingerprint_of_snapshot(entry.storage) ==
    entry.storage_fingerprint`` holds unless the registry entry was hand-edited."""
    provider = snapshot.get("provider")
    if not isinstance(provider, str):
        return ""
    options = {k: v for k, v in snapshot.items() if k != "provider"}
    return _identity_hash(provider, options)


def resolve_store_config(workspace: Workspace) -> StorageConfig:
    """The effective :class:`StorageConfig` for opening ``workspace``.

    For a **registered** workspace the non-secret identity comes from the registry
    entry's snapshot (authoritative and self-contained — the store opens even if
    ``config.toml`` was edited/deleted); only *secret* options are merged in from
    the entry's named ``config.toml`` template (or the provider SDK's own
    credential chain when the template is gone). ``config.toml`` never overrides the
    non-secret identity.

    An **unregistered** workspace (a raw ``Workspace(root=…)``, or one being
    created before its entry is written) resolves the ``"default"`` service — the
    bundled local-disk store with zero config."""
    from . import registry

    entry = registry.get_by_root(workspace.root)  # local read, store-free
    if entry is None:
        return load_storage_config(workspace, DEFAULT_STORAGE_SERVICE)
    provider = entry.storage.get("provider") if isinstance(entry.storage, dict) else None
    if not isinstance(provider, str) or not provider.strip():
        # Legacy/empty snapshot: fall back to the named template in config.
        return load_storage_config(workspace, entry.storage_service or DEFAULT_STORAGE_SERVICE)
    non_secret = {k: v for k, v in entry.storage.items() if k != "provider"}
    return StorageConfig(
        provider=provider,
        root=workspace.root,
        options={**non_secret, **_secret_options(workspace, entry.storage_service)},
    )


def _secret_options(workspace: Workspace, service: str) -> dict[str, Any]:
    """The secret-hinted options of ``service``'s ``config.toml`` template (empty
    when the template is absent — the provider may still find creds via env/SDK)."""
    try:
        cfg = load_storage_config(workspace, service or DEFAULT_STORAGE_SERVICE)
    except StorageConfigInvalid:
        return {}
    return {
        k: v for k, v in cfg.options.items() if any(hint in k.lower() for hint in _SECRET_HINTS)
    }
