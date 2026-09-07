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

"""Workspace ids: what makes one valid, and minting them.

A ``workspace_id`` is a workspace's stable name: 3-40 characters from
``[a-z0-9_-]``, starting with a letter or digit. It survives a directory rename, and
is how ``--workspace`` addresses a workspace held in the machine's store of
workspaces. Minted ids are ``ws_`` + 16 lowercase base32 characters (80 bits from
:func:`secrets`); a caller-supplied one (``workspace create --id``) may be any valid
id, so ``my-workspace`` is as legitimate as ``ws_qf7imkc7f6oqzfwt``.

Validity is all this module decides. Whether a given ``--workspace`` argument means an
id or a filesystem path is :meth:`dgml_core.storage.Workspace._from_workspaces_store`'s
question, because — ids no longer carrying a distinguishing prefix — answering it needs
the store of workspaces, which this module deliberately does not know about.

Its own module, depending on nothing, deliberately: id minting is needed by the
workspaces store, by the CLI, *and* by :mod:`dgml_core.migrations` (which backfills
an id into a pre-id workspace). Leaving it in the store's module would make a
migration that never touches the store import one anyway.
"""

from __future__ import annotations

import base64
import re
import secrets
from typing import Protocol

#: Prefix on every id :func:`new_workspace_id` mints. No longer *required* of an id —
#: a caller-supplied one need not carry it — and no longer meaningful to resolution;
#: it survives because it makes a minted id self-describing at a glance.
ID_PREFIX = "ws_"

#: What a workspace id may look like. Lowercase because the local store of workspaces
#: uses an id as a **directory name**, and a case-insensitive filesystem would let
#: ``Notes`` and ``notes`` collide; no dot, separator or whitespace so an id is always a
#: safe single path segment; anchored at both ends so a path that merely *contains* an
#: id is not one.
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,39}\Z")

#: The rule in :data:`_ID_RE`, in words. One source of truth for every message that has
#: to tell a caller why their id was rejected.
ID_SHAPE = "3-40 characters from [a-z0-9_-], starting with a lowercase letter or digit"


def new_workspace_id() -> str:
    """A fresh opaque workspace id: ``ws_`` + 16 lowercase base32 chars (80 bits).

    Non-semantic (survives a directory rename) and hyphen/separator-free. Not
    collision-checked — use :func:`mint_workspace_id` when assigning an id to a
    workspace."""
    slug = base64.b32encode(secrets.token_bytes(10)).decode("ascii").lower().rstrip("=")
    return f"{ID_PREFIX}{slug}"


class SupportsExists(Protocol):
    """The one thing :func:`mint_workspace_id` needs of a store of workspaces.

    A structural type rather than an import, so this module keeps its "depends on
    nothing" property and a migration that mints an id never pulls a store in."""

    def exists(self, workspace_id: str) -> bool: ...


def mint_workspace_id(store: SupportsExists | None = None) -> str:
    """A fresh workspace id, re-rolled while ``store`` already holds it.

    80 bits from :func:`secrets` will not collide in practice; the re-roll is
    belt-and-suspenders against two workspaces sharing an id and shadowing each other
    at ``--workspace <id>``.

    Passing a ``store`` makes the check **authoritative and complete** for workspaces
    that store lists — including ones created on another machine, when the store is
    shared. Without one it is an unchecked mint, which is the honest answer for a
    detached workspace or an id backfilled by a migration: there is no list to consult.
    A store that can be raced (two machines minting in the same instant) should also
    make its insert conditional, since no pre-check can close that window."""
    wid = new_workspace_id()
    if store is None:
        return wid
    while store.exists(wid):
        wid = new_workspace_id()
    return wid


def is_workspace_id(value: str) -> bool:
    """Whether ``value`` is a **valid** workspace id (:data:`ID_SHAPE`).

    Not "is this an id rather than a path" — a bare slug is a legal relative directory
    name too, so that question needs the store of workspaces and is answered by
    :meth:`dgml_core.storage.Workspace._from_workspaces_store`. What this test *does*
    guarantee is that anything carrying a separator, a dot, an uppercase letter or
    leading ``./`` is not an id at all, which is what lets that resolution start with a
    cheap store-free rejection — and what keeps ``./notes`` addressing the directory even
    when a workspace ``notes`` is listed."""
    return _ID_RE.match(value) is not None
