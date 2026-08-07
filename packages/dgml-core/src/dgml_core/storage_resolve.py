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

"""Resolving a workspace's storage backend from configuration.

This is DGML's store *resolver*, kept separate from the store *abstraction*
(:mod:`dgml_core.storage_service`, the ``StorageService`` ABC a third party
implements). Nothing here is part of the "implement a store" surface; it is how
DGML turns configuration into a live, identified store:

- **Read** — :func:`load_storage_config` resolves a named ``[storage.<name>]``
  template from the merged ``config.toml`` into a :class:`StorageConfig`.
- **Build** — :func:`make_store` resolves the ``provider`` dotted path to its
  :class:`StorageService` subclass and constructs it.
- **Identify** — :func:`storage_fingerprint` / :func:`storage_snapshot` /
  :func:`fingerprint_of_snapshot` produce the credential-free store identity the
  per-machine registry records and seals; :func:`secret_options` is the
  credential complement of the snapshot, merged back in only at open time.

Deciding *which* config a given (possibly registered) workspace opens with lives
one layer up, in :func:`dgml_core.registry.resolve_store_config`, because it
consults the registry — this module has no knowledge of the registry.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import load_merged_config
from .errors import StorageConfigInvalid, StorageProviderUnresolvable
from .models_config import ConfigSection
from .storage import Workspace
from .storage_service import StorageConfig, StorageService

# The bundled default: local disk. Used when the config has no ``storage``
# section. Resolved through the same path as any third-party provider.
DEFAULT_STORAGE_PROVIDER = "dgml_core.storage_local:LocalStore"

# The storage-service name a workspace uses when none is chosen at create time,
# and the name a bare (unnamed) ``[storage]`` table resolves as.
DEFAULT_STORAGE_SERVICE = "default"

# Option keys never folded into the store fingerprint / snapshot — rotating a
# credential must not read as "the store moved", and secrets never reach the
# plaintext registry.
_SECRET_HINTS = ("key", "secret", "token", "password", "credential")


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
    :func:`fingerprint_of_snapshot`; :func:`secret_options` is its complement."""
    snapshot: dict[str, Any] = {"provider": config.provider}
    snapshot.update(
        (k, v)
        for k, v in config.options.items()
        if not any(hint in k.lower() for hint in _SECRET_HINTS)
    )
    return snapshot


def secret_options(config: StorageConfig) -> dict[str, Any]:
    """The **secret-hinted** options of a store config — the complement of
    :func:`storage_snapshot`. Merged back into a registered workspace's non-secret
    snapshot at open time; never persisted to the registry."""
    return {
        k: v for k, v in config.options.items() if any(hint in k.lower() for hint in _SECRET_HINTS)
    }


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
