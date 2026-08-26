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

"""Tests for the per-machine workspace index + id-or-path addressing.

The index records where workspaces were last seen; what backend each one uses lives
in its own ``config.toml`` and is covered by ``test_workspace_config.py``.

The autouse ``_isolate_user_config`` fixture (conftest) points
``XDG_CONFIG_HOME`` at a per-test tmp dir, so ``registry_path()`` is sandboxed and
the developer's real ``~/.config/dgml/workspaces.json`` is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core import registry
from dgml_core.errors import CorruptMetadata
from dgml_core.registry import RegistryEntry
from dgml_core.storage import Workspace


def _entry(workspace_id: str, root: Path, *, name: str = "W", org: str = "acme") -> RegistryEntry:
    return RegistryEntry(
        workspace_id=workspace_id,
        name=name,
        organization=org,
        root=str(root),
        created_at="2026-08-05T12:00:00Z",
        schema_version=1,
    )


# ---------------------------------------------------------------- new_workspace_id


def test_new_workspace_id_shape_and_uniqueness() -> None:
    ids = {registry.new_workspace_id() for _ in range(50)}
    assert len(ids) == 50  # no collisions in a small sample
    for i in ids:
        assert i.startswith("ws_")
        assert len(i) == 19  # "ws_" + 16 base32 chars
        assert i[3:].isalnum() and i[3:].islower()


def test_mint_workspace_id_skips_a_registered_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mint re-rolls when the freshly-generated id already exists in the registry."""
    taken = "ws_takentakentaken1"
    registry.register(_entry(taken, tmp_path / "a"))

    # First roll collides with the registered id, second is free.
    rolls = iter([taken, "ws_freefreefreefre1"])
    monkeypatch.setattr(registry, "new_workspace_id", lambda: next(rolls))
    assert registry.mint_workspace_id() == "ws_freefreefreefre1"


# ---------------------------------------------------------------- registry I/O


def test_read_registry_absent_is_empty(tmp_path: Path) -> None:
    assert not registry.registry_path().exists()
    assert registry.read_registry() == {}
    assert registry.list_entries() == []
    assert registry.get("ws_nope") is None


def test_register_get_list_remove_roundtrip(tmp_path: Path) -> None:
    a = _entry("ws_aaaaaaaaaaaaaaaa", tmp_path / "a", name="A")
    b = _entry("ws_bbbbbbbbbbbbbbbb", tmp_path / "b", name="B")
    registry.register(a)
    registry.register(b)

    assert registry.get("ws_aaaaaaaaaaaaaaaa") == a
    assert {e.workspace_id for e in registry.list_entries()} == {a.workspace_id, b.workspace_id}
    # list is id-sorted (stable output)
    assert [e.workspace_id for e in registry.list_entries()] == [a.workspace_id, b.workspace_id]

    assert registry.remove("ws_aaaaaaaaaaaaaaaa") is True
    assert registry.get("ws_aaaaaaaaaaaaaaaa") is None
    assert registry.remove("ws_aaaaaaaaaaaaaaaa") is False  # already gone


def test_register_is_idempotent_upsert(tmp_path: Path) -> None:
    registry.register(_entry("ws_cccccccccccccccc", tmp_path / "c", name="Old"))
    registry.register(_entry("ws_cccccccccccccccc", tmp_path / "c", name="New"))
    entries = registry.list_entries()
    assert len(entries) == 1
    assert entries[0].name == "New"


def test_get_by_root_reverse_lookup(tmp_path: Path) -> None:
    registry.register(_entry("ws_dddddddddddddddd", tmp_path / "d"))
    hit = registry.get_by_root(tmp_path / "d")
    assert hit is not None and hit.workspace_id == "ws_dddddddddddddddd"
    assert registry.get_by_root(tmp_path / "nowhere") is None


def test_corrupt_registry_raises(tmp_path: Path) -> None:
    p = registry.registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        registry.read_registry()


def test_non_object_registry_raises(tmp_path: Path) -> None:
    p = registry.registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        registry.read_registry()


# ---------------------------------------------------------------- id-or-path resolution


def test_resolve_by_id_uses_registered_root(tmp_path: Path) -> None:
    root = (tmp_path / "acme-ws").resolve()
    wid = registry.new_workspace_id()
    registry.register(_entry(wid, root))
    assert Workspace.resolve(wid).root == root


def test_resolve_unregistered_token_is_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bare token that is NOT a registered id resolves as a (relative) path.
    monkeypatch.chdir(tmp_path)
    assert Workspace.resolve("some_dir").root == (tmp_path / "some_dir").resolve()


def test_resolve_absolute_path_is_never_an_id(tmp_path: Path) -> None:
    p = tmp_path / "ws"
    assert Workspace.resolve(p).root == p.resolve()
    assert Workspace.resolve(str(p)).root == p.resolve()


def test_resolve_path_typed_id_round_trips(tmp_path: Path) -> None:
    # --workspace is argparse type=Path, so an id arrives as Path("ws_...").
    root = (tmp_path / "ws").resolve()
    wid = registry.new_workspace_id()
    registry.register(_entry(wid, root))
    assert Workspace.resolve(Path(wid)).root == root


def test_workspace_constructed_by_root_has_no_id(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    assert ws.workspace_id is None  # no registry / no workspace.json required


# ---------------------------------------------------- legacy rows / self-healing


def test_from_dict_tolerates_legacy_storage_keys(tmp_path: Path) -> None:
    """A pre-upgrade row carried the binding inline. Reading one must not raise —
    those keys are simply not this dataclass's business any more."""
    entry = RegistryEntry.from_dict(
        "ws_legacyxxxxxxxxxx",
        {
            "name": "W",
            "organization": "acme",
            "root": str(tmp_path / "ws"),
            "storage_service": "svcA",
            "storage": {"blobs": {"provider": "x:Y"}, "docs": {"provider": "x:Y"}},
            "storage_fingerprint": "sha256:deadbeef",
            "created_at": "2026-08-05T12:00:00Z",
            "schema_version": 1,
        },
    )
    assert entry.name == "W"
    assert entry.root == str(tmp_path / "ws")
    assert not any(f.startswith("storage") for f in entry.to_dict())


def test_index_write_drops_legacy_storage_keys(tmp_path: Path) -> None:
    """Rewriting a legacy row leaves no second, powerless copy of the binding."""
    import json

    wid = "ws_legacyxxxxxxxxxx"
    root = tmp_path / "ws"
    registry.registry_path().parent.mkdir(parents=True, exist_ok=True)
    registry.registry_path().write_text(
        json.dumps({wid: {"name": "W", "root": str(root), "storage": {"blobs": {}}}})
    )
    registry.index_workspace(Workspace(root=root), workspace_id=wid, name="W", organization="acme")
    written = json.loads(registry.registry_path().read_text())[wid]
    assert not any(k.startswith("storage") for k in written)


def test_ensure_registered_corrects_a_stale_root(tmp_path: Path) -> None:
    """A moved workspace re-points its own row on open — the reason an explicit
    `workspace register` is no longer needed."""
    from dgml_core import workspace_config

    old_root, new_root = tmp_path / "before", tmp_path / "after"
    new_root.mkdir()
    wid = "ws_movedxxxxxxxxxxx"
    registry.register(_entry(wid, old_root))

    ws = Workspace(root=new_root)
    workspace_config.write_identity(ws, workspace_id=wid, organization="acme")
    registry.ensure_registered(ws)

    entry = registry.get(wid)
    assert entry is not None
    assert entry.root == str(new_root)
    assert registry.get_by_root(old_root) is None


def test_ensure_registered_does_not_rewrite_when_root_matches(tmp_path: Path) -> None:
    """The common path runs on every command, so it must not touch the file."""
    from dgml_core import workspace_config

    root = tmp_path / "ws"
    root.mkdir()
    wid = "ws_stablexxxxxxxxxx"
    registry.register(_entry(wid, root))
    ws = Workspace(root=root)
    workspace_config.write_identity(ws, workspace_id=wid, organization="acme")

    before = registry.registry_path().stat().st_mtime_ns
    registry.ensure_registered(ws)
    assert registry.registry_path().stat().st_mtime_ns == before


def test_ensure_registered_is_store_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a complete identity block — what `workspace create` writes — indexing a
    remote-backed workspace costs no round trip and works while its backend is
    unreachable. This is the payoff of keeping identity out of the store."""
    from dgml_core import workspace_config

    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace(root=root)
    workspace_config.write_identity(
        ws, workspace_id="ws_remotexxxxxxxxxx", name="W", organization="acme"
    )

    def explode(self: Workspace) -> None:
        raise AssertionError("ensure_registered must not open the store")

    monkeypatch.setattr(Workspace, "docs", property(explode))
    registry.ensure_registered(ws)

    entry = registry.get("ws_remotexxxxxxxxxx")
    assert entry is not None and entry.organization == "acme"


# ------------------------------------------------- the external-config hint


def _external(tmp_path: Path) -> Path:
    cfg = tmp_path / "elsewhere" / "acme.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text('[storage]\nprovider = "dgml_core.storage_local:LocalStore"\n')
    return cfg


def test_config_path_is_recorded_only_when_external(tmp_path: Path) -> None:
    """The default `<root>/config.toml` is derivable from `root`; recording it would be
    noise that also has to be kept in step on every move."""
    root = tmp_path / "ws"
    root.mkdir()

    registry.index_workspace(
        Workspace(root=root), workspace_id="ws_defaultxxxxxxxxx", name="W", organization="acme"
    )
    assert registry.get("ws_defaultxxxxxxxxx").config_path is None  # type: ignore[union-attr]

    cfg = _external(tmp_path)
    registry.index_workspace(
        Workspace(root=root, config_override=cfg),
        workspace_id="ws_externalxxxxxxxx",
        name="W",
        organization="acme",
    )
    assert registry.get("ws_externalxxxxxxxx").config_path == str(cfg)  # type: ignore[union-attr]


def test_recorded_config_is_used_when_the_default_is_absent(tmp_path: Path) -> None:
    """The point of the hint: a workspace created with --workspace-config opens by id
    without having to repeat the flag every time."""
    root = tmp_path / "ws"
    root.mkdir()
    cfg = _external(tmp_path)
    registry.index_workspace(
        Workspace(root=root, config_override=cfg),
        workspace_id="ws_hintxxxxxxxxxxxx",
        name="W",
        organization="acme",
    )
    assert Workspace.resolve("ws_hintxxxxxxxxxxxx").config_path == cfg
    assert Workspace.resolve(root).config_path == cfg  # by path, too


def test_an_in_workspace_config_always_wins_over_the_hint(tmp_path: Path) -> None:
    """The hint is consulted only when `<root>/config.toml` is absent, so it can never
    redirect a workspace that has a perfectly good config of its own."""
    root = tmp_path / "ws"
    root.mkdir()
    own = root / "config.toml"
    own.write_text('[storage]\nprovider = "dgml_core.storage_local:LocalStore"\n')
    registry.register(
        RegistryEntry(
            workspace_id="ws_hintxxxxxxxxxxxx",
            name="W",
            organization="acme",
            root=str(root),
            created_at="2026-08-05T12:00:00Z",
            schema_version=1,
            config_path=str(_external(tmp_path)),
        )
    )
    assert Workspace.resolve(root).config_path == own


def test_a_stale_hint_is_ignored(tmp_path: Path) -> None:
    """A hint pointing at a file that is gone degrades to the ordinary missing-config
    path, never to a confusing half-resolution."""
    root = tmp_path / "ws"
    root.mkdir()
    registry.register(
        RegistryEntry(
            workspace_id="ws_stalexxxxxxxxxxx",
            name="W",
            organization="acme",
            root=str(root),
            created_at="2026-08-05T12:00:00Z",
            schema_version=1,
            config_path=str(tmp_path / "deleted.toml"),
        )
    )
    assert Workspace.resolve(root).config_override is None
    assert Workspace.resolve(root).config_path == root / "config.toml"


def test_ensure_registered_refreshes_a_changed_config_path(tmp_path: Path) -> None:
    """Same self-healing rule as `root`: where a workspace was last opened is the best
    guess for where to find it next."""
    from dgml_core import workspace_config

    root = tmp_path / "ws"
    root.mkdir()
    cfg = _external(tmp_path)
    ws = Workspace(root=root, config_override=cfg)
    workspace_config.write_identity(
        ws, workspace_id="ws_movecfgxxxxxxxxx", name="W", organization="acme"
    )
    registry.ensure_registered(ws)
    assert registry.get("ws_movecfgxxxxxxxxx").config_path == str(cfg)  # type: ignore[union-attr]

    # The config moves into the workspace: the hint is dropped, not left dangling.
    (root / "config.toml").write_text(cfg.read_text())
    workspace_config.write_identity(
        Workspace(root=root), workspace_id="ws_movecfgxxxxxxxxx", name="W", organization="acme"
    )
    registry.ensure_registered(Workspace(root=root))
    assert registry.get("ws_movecfgxxxxxxxxx").config_path is None  # type: ignore[union-attr]
