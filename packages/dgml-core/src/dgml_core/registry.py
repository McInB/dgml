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

The index is **per-machine** state, deliberately separate from the per-workspace
``workspace.json`` (which travels with the directory): the same workspace opened on
two machines has one id but two entries with different roots. It is machine-managed
(JSON, like every other metadata file — ``workspace.json``, ``docset.json``), not
hand-edited like ``config.toml``.

**This file is a regenerable cache and nothing more.** It records where workspaces
were last seen so they can be listed and opened by id; it says nothing about *how* a
workspace stores its data. That binding lives in the workspace's own ``config.toml``
(:mod:`dgml_core.workspace_config`), which is authoritative and travels with the
workspace. Deleting this file loses only the ability to enumerate — every entry comes
back as each workspace is next opened by path (:func:`ensure_registered`), with its
``workspace_id`` intact.

Because the entry carries no authority, it is also self-healing: opening a workspace
that has moved corrects its recorded ``root`` in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import Workspace, read_json, user_config_path, write_json_atomic

# Id minting lives in its own module (see :mod:`dgml_core.workspace_id`) because the
# workspaces store and the migrations both need it without needing each other. The
# redundant-alias spelling is an explicit re-export, for the callers that still reach
# for these names here (and so ``mypy --strict`` accepts it).
from .workspace_id import ID_PREFIX as ID_PREFIX
from .workspace_id import is_workspace_id as is_workspace_id
from .workspace_id import new_workspace_id as new_workspace_id

REGISTRY_FILE = "workspaces.json"


def registry_path() -> Path:
    """The registry file, next to the user ``config.toml`` (honors
    ``XDG_CONFIG_HOME``/``APPDATA``)."""
    return user_config_path().parent / REGISTRY_FILE


def mint_workspace_id() -> str:
    """A fresh workspace id, re-rolled if this machine's index already holds it.

    80 bits from :func:`secrets` won't collide in practice; the re-roll is
    belt-and-suspenders against two workspaces sharing an id and shadowing each other
    at ``--workspace <id>``. **Best-effort only** — the index is a regenerable cache,
    so it may not list every workspace on the machine, and never lists one from
    another machine."""
    wid = new_workspace_id()
    while get(wid) is not None:
        wid = new_workspace_id()
    return wid


@dataclass(frozen=True)
class RegistryEntry:
    """One workspace's row in the index: where it was last seen, and what to call it.

    ``root`` is the local directory the workspace was opened at. ``name`` and
    ``organization`` are copies carried so ``dgml workspace list`` can render a row
    without opening (and possibly failing to reach) every workspace's store.

    ``config_path`` is recorded **only** when the workspace's ``config.toml`` lives
    outside ``root`` (``--workspace-config``); absent means the default
    ``<root>/config.toml``, which is derivable and would be noise to store. It is a
    **hint, not an address**: resolution consults it only when the default is missing,
    and ignores it when it points at a file that is gone. Nothing here is authoritative
    — a stale hint degrades to the same "config is missing" error you would get
    without it, never to opening the wrong config.
    """

    workspace_id: str
    name: str
    organization: str
    root: str | None
    created_at: str
    schema_version: int
    config_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "organization": self.organization,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        if self.root is not None:
            d["root"] = self.root
        if self.config_path is not None:
            d["config_path"] = self.config_path
        return d

    @classmethod
    def from_dict(cls, workspace_id: str, data: dict[str, Any]) -> RegistryEntry:
        """Parse one row, ignoring unknown keys.

        Tolerating extras is what upgrades a pre-existing ``workspaces.json`` for
        free: the ``storage`` / ``storage_service`` / ``storage_fingerprint`` an older
        dgml wrote are read as noise here and dropped the next time the row is
        rewritten. The migration reads them from the raw JSON before that happens (see
        :func:`dgml_core.migrations.migrate_workspace_config`)."""
        return cls(
            workspace_id=workspace_id,
            name=str(data.get("name", "")),
            organization=str(data.get("organization", "")),
            root=data.get("root"),
            created_at=str(data.get("created_at", "")),
            schema_version=int(data["schema_version"])
            if isinstance(data.get("schema_version"), int)
            else 0,
            config_path=data.get("config_path")
            if isinstance(data.get("config_path"), str)
            else None,
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
    happen only at ``workspace create``, at first-open indexing, and when a moved
    workspace's root is corrected, so the (non-cross-process-atomic) RMW is
    acceptable — an interleaved lost update self-heals on the next open, since this
    is an idempotent upsert and the row it writes is a cache."""
    data = _read_raw()
    data[entry.workspace_id] = entry.to_dict()
    write_json_atomic(registry_path(), data)


def get(workspace_id: str) -> RegistryEntry | None:
    entry = _read_raw().get(workspace_id)
    return RegistryEntry.from_dict(workspace_id, entry) if isinstance(entry, dict) else None


def get_by_root(root: Path) -> RegistryEntry | None:
    """The entry whose local ``root`` is ``root`` (path addressing / open-by-path).

    Deterministic on the off chance two ids share a root: lowest id wins. Reads the
    index once rather than once per candidate — it is consulted on the config-recovery
    path, where a workspace with many siblings would otherwise pay a file read each."""
    target = root.resolve()
    raw = _read_raw()
    for wid in sorted(raw):
        data = raw.get(wid)
        if not isinstance(data, dict):
            continue
        entry = RegistryEntry.from_dict(wid, data)
        if entry.root is not None and Path(entry.root).resolve() == target:
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


def raw_entry_by_root(root: Path) -> dict[str, Any] | None:
    """The **unparsed** row whose ``root`` is ``root``, or ``None``.

    Exists for the one caller that needs fields :class:`RegistryEntry` deliberately
    drops: :func:`dgml_core.migrations.migrate_workspace_config` reads a pre-upgrade
    row's ``storage`` snapshot out of it. Everything else should use :func:`get_by_root`.
    """
    target = root.resolve()
    for wid in sorted(_read_raw()):
        data = _read_raw().get(wid)
        if not isinstance(data, dict):
            continue
        entry_root = data.get("root")
        if isinstance(entry_root, str) and Path(entry_root).resolve() == target:
            return {**data, "workspace_id": wid}
    return None


# ------------------------------------------------------- indexing a workspace


def external_config_path(ws: Workspace) -> str | None:
    """``ws``'s config path when it lives outside the workspace, else ``None``.

    The default ``<root>/config.toml`` is derivable from ``root``, so recording it
    would be noise — and would have to be kept in step with ``root`` on every move."""
    return str(ws.config_path) if ws.config_override is not None else None


def index_workspace(ws: Workspace, *, workspace_id: str, name: str, organization: str) -> None:
    """Upsert ``ws``'s row, stamping ``created_at`` and the current schema version —
    the one place a row is written.

    Store-free: everything it records is either passed in or read from ``ws.root``, so
    it is safe to call before a workspace's store has ever been built."""
    from .errors import now_iso
    from .migrations import WORKSPACE_SCHEMA_VERSION

    register(
        RegistryEntry(
            workspace_id=workspace_id,
            name=name,
            organization=organization,
            root=str(ws.root),
            created_at=now_iso(),
            schema_version=WORKSPACE_SCHEMA_VERSION,
            config_path=external_config_path(ws),
        )
    )


def ensure_registered(ws: Workspace) -> None:
    """Index ``ws`` if it is missing, or correct its recorded location if it changed.

    Called on every command, so the common path must not write: an entry whose ``root``
    and ``config_path`` both already match is left alone, and a workspace with no id yet
    (one is minted by the backfill migration on first open) is skipped entirely.

    Correcting the location in place is safe *because the row carries no authority*.
    Under the old registry an overwrite would have re-snapshotted the workspace's
    storage, so a stale root could only be fixed by an explicit command; now the row is
    a cache and where a workspace opens from is decided by its own ``config.toml``.

    The ``config_path`` hint follows the same rule as ``root``: the location a workspace
    was last opened at is the best guess for where to find it next. Recording it from an
    ordinary open is harmless even when the config was a one-off, because it is only
    ever consulted when the default is missing, and a config naming a different backend
    is caught by the seal rather than opened silently.

    Identity is read from that same ``config.toml`` when present, falling back to
    ``workspace.json`` — which matters for a remote-backed workspace, where the
    fallback is a network round trip on every command, and fails outright when the
    backend is unreachable."""
    from . import workspace_config

    identity = workspace_config.read_identity(ws)
    wid = identity.workspace_id or ws.workspace_id
    if wid is None:
        return
    existing = get(wid)
    if existing is not None and existing.root is not None:
        unchanged = Path(existing.root) == ws.root and existing.config_path == (
            external_config_path(ws)
        )
        if unchanged:
            return
        # Moved (or opened from a different config): keep the recorded identity, just
        # re-point it.
        index_workspace(
            ws, workspace_id=wid, name=existing.name, organization=existing.organization
        )
        return
    index_workspace(
        ws,
        workspace_id=wid,
        name=identity.name or ws.display_name,
        organization=identity.organization or ws.organization,
    )
