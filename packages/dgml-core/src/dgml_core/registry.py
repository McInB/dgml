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

"""The per-machine workspace registry.

A workspace has a stable ``workspace_id`` (minted at ``dgml workspace create``,
carried in ``workspace.json``). This module maintains a small JSON index —
``~/.config/dgml/workspaces.json`` (sibling of the user ``config.toml``) — mapping
each ``workspace_id`` to where that workspace lives, so it can be opened by id
(``dgml --workspace <id>``) and listed (``dgml workspace list``).

The registry is **per-machine** state, deliberately separate from the
per-workspace ``workspace.json`` (which travels with the directory): the same
workspace opened on two machines has one id but two registry entries with
different roots. It is machine-managed (JSON, like every other metadata file —
``workspace.json``, ``docset.json``), not hand-edited like ``config.toml``.

Today only ``LocalStore`` ships, so every entry records a local ``root`` and is
opened through it. The ``storage`` identity + ``storage_fingerprint`` are recorded
now (they cost nothing to compute) so the deferred store-mismatch guard is a small
follow-up; nothing reads them yet. Reconstructing a *remote* store from
``entry.storage`` (open-by-id with no local root) lands with the remote store.
"""

from __future__ import annotations

import base64
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import Workspace, read_json, user_config_path, write_json_atomic

REGISTRY_FILE = "workspaces.json"
_ID_PREFIX = "ws_"


def registry_path() -> Path:
    """The registry file, next to the user ``config.toml`` (honors
    ``XDG_CONFIG_HOME``/``APPDATA``)."""
    return user_config_path().parent / REGISTRY_FILE


def new_workspace_id() -> str:
    """A fresh opaque workspace id: ``ws_`` + 16 lowercase base32 chars (80 bits).

    Non-semantic (survives a directory rename) and hyphen/separator-free — the
    ``ws_`` prefix lets ``Workspace.resolve`` tell an id from a path without a
    dedicated flag. Not collision-checked — use :func:`mint_workspace_id` when
    assigning an id to a workspace."""
    slug = base64.b32encode(secrets.token_bytes(10)).decode("ascii").lower().rstrip("=")
    return f"{_ID_PREFIX}{slug}"


def mint_workspace_id() -> str:
    """A fresh workspace id guaranteed not to already be in this machine's registry.

    80 bits from :func:`secrets` won't collide in practice; the registry re-roll is
    belt-and-suspenders so two workspaces can never share an id (and shadow each
    other on open)."""
    wid = new_workspace_id()
    while get(wid) is not None:
        wid = new_workspace_id()
    return wid


@dataclass(frozen=True)
class RegistryEntry:
    """One workspace's row in the registry.

    ``root`` is the local store location (always set today, since only
    ``LocalStore`` ships); ``storage`` is the store *identity* (provider +
    non-secret options) and ``storage_fingerprint`` its credential-free hash —
    both recorded for the deferred store-mismatch guard.
    """

    workspace_id: str
    name: str
    organization: str
    root: str | None
    storage: dict[str, Any]
    storage_fingerprint: str
    created_at: str
    schema_version: int

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "organization": self.organization,
            "storage": self.storage,
            "storage_fingerprint": self.storage_fingerprint,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        if self.root is not None:
            d["root"] = self.root
        return d

    @classmethod
    def from_dict(cls, workspace_id: str, data: dict[str, Any]) -> RegistryEntry:
        return cls(
            workspace_id=workspace_id,
            name=str(data.get("name", "")),
            organization=str(data.get("organization", "")),
            root=data.get("root"),
            storage=data.get("storage", {}) if isinstance(data.get("storage"), dict) else {},
            storage_fingerprint=str(data.get("storage_fingerprint", "")),
            created_at=str(data.get("created_at", "")),
            schema_version=int(data["schema_version"])
            if isinstance(data.get("schema_version"), int)
            else 0,
        )


def _read_raw() -> dict[str, Any]:
    """The registry as a raw ``{id: entry-dict}`` mapping (``{}`` when absent).

    Raises :class:`~dgml_core.errors.CorruptMetadata` on malformed JSON /
    duplicate ids (via :func:`read_json`) or a non-object top level."""
    from .errors import CorruptMetadata

    path = registry_path()
    if not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise CorruptMetadata(f"{path} must contain a JSON object of workspace_id -> entry")
    return data


def read_registry() -> dict[str, RegistryEntry]:
    """Every registered workspace, keyed by ``workspace_id``."""
    return {wid: RegistryEntry.from_dict(wid, entry) for wid, entry in _read_raw().items()}


def register(entry: RegistryEntry) -> None:
    """Insert or replace ``entry`` (idempotent upsert by id), atomically.

    Whole-file read-modify-write; each write is atomic (write-temp-rename). Writes
    happen only at create / ``workspace register`` / first-open registration, so
    the (non-cross-process-atomic) RMW is acceptable — an interleaved lost update
    self-heals on the next open, since register is idempotent."""
    data = _read_raw()
    data[entry.workspace_id] = entry.to_dict()
    write_json_atomic(registry_path(), data)


def get(workspace_id: str) -> RegistryEntry | None:
    entry = _read_raw().get(workspace_id)
    return RegistryEntry.from_dict(workspace_id, entry) if isinstance(entry, dict) else None


def get_by_root(root: Path) -> RegistryEntry | None:
    """The entry whose local ``root`` is ``root`` (path addressing / open-by-path).

    Deterministic on the off chance two ids share a root: lowest id wins."""
    target = root.resolve()
    for wid in sorted(_read_raw()):
        entry = get(wid)
        if entry is not None and entry.root is not None and Path(entry.root).resolve() == target:
            return entry
    return None


def list_entries() -> list[RegistryEntry]:
    """All entries, sorted by id (stable output for ``dgml workspace list``)."""
    reg = read_registry()
    return [reg[wid] for wid in sorted(reg)]


def remove(workspace_id: str) -> bool:
    """Drop ``workspace_id`` from the registry. Returns whether it was present."""
    data = _read_raw()
    if workspace_id not in data:
        return False
    del data[workspace_id]
    write_json_atomic(registry_path(), data)
    return True


# --------------------------------------------------- building entries from a workspace


def entry_for(
    ws: Workspace,
    *,
    name: str,
    organization: str,
    workspace_id: str,
    created_at: str,
    schema_version: int,
) -> RegistryEntry:
    """Build the registry entry for ``ws``: its local ``root`` plus the store
    *identity* (``provider`` — LocalStore carries no options) and a
    credential-free ``storage_fingerprint``, recorded for the deferred guard.

    (Lazy imports keep ``registry`` free of a top-level ``storage_service`` /
    ``migrations`` cycle.)"""
    from .storage_service import load_storage_config, storage_fingerprint

    cfg = load_storage_config(ws)
    return RegistryEntry(
        workspace_id=workspace_id,
        name=name,
        organization=organization,
        root=str(ws.root),
        storage={"provider": cfg.provider},  # remote will extend this (non-secret conn info)
        storage_fingerprint=storage_fingerprint(cfg),
        created_at=created_at,
        schema_version=schema_version,
    )


def _put_entry(ws: Workspace, *, workspace_id: str, name: str, organization: str) -> None:
    """Build ``ws``'s registry entry (stamping ``created_at`` / current schema
    version) and upsert it — the one place the entry is assembled and written."""
    from .errors import now_iso
    from .migrations import WORKSPACE_SCHEMA_VERSION

    register(
        entry_for(
            ws,
            name=name,
            organization=organization,
            workspace_id=workspace_id,
            created_at=now_iso(),
            schema_version=WORKSPACE_SCHEMA_VERSION,
        )
    )


def register_workspace(
    ws: Workspace, *, name: str | None = None, organization: str | None = None
) -> str:
    """Register ``ws`` on this machine, returning its ``workspace_id``.

    Mints an id and writes it back into ``workspace.json`` when the workspace lacks
    one (so the directory self-describes), then upserts the registry entry. This is
    the *authoritative* register — a repeat call re-points the recorded ``root`` (the
    moved-directory fix), unlike the additive :func:`ensure_registered`.
    ``name``/``organization`` default to the workspace's own identity."""
    name = ws.display_name if name is None else name
    organization = ws.organization if organization is None else organization
    wid = ws.workspace_id
    if wid is None:
        wid = mint_workspace_id()
        ws.write_meta(name=name, organization=organization, workspace_id=wid)
    _put_entry(ws, workspace_id=wid, name=name, organization=organization)
    return wid


def ensure_registered(ws: Workspace) -> None:
    """Add ``ws`` to this machine's registry if it has an id and isn't indexed yet.

    Idempotent and additive: never overwrites an existing entry (that is what
    :func:`register_workspace` / the explicit ``dgml workspace register`` does). A
    no-op for a workspace with no ``workspace_id`` (one is minted by the backfill
    migration on first open)."""
    wid = ws.workspace_id
    if wid is None or get(wid) is not None:
        return
    _put_entry(ws, workspace_id=wid, name=ws.display_name, organization=ws.organization)
