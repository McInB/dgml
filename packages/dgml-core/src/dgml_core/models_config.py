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

"""The ``[models]`` tier block — the simplified model entry point.

Four tiers, cheapest to strongest, each mapped to a set of tasks:

* ``light`` ...... classification, style
* ``standard`` ... transcription, text extraction
* ``advanced`` ... labeling, value extraction
* ``expert`` ..... schema generation

Per-task fields (``generation.label_model`` etc.) are optional *overrides* that
win over the tier; when a task names no model of its own it falls back to its
tier here. Each tier also carries an optional ``<tier>_api_key_env`` and
``<tier>_api_base`` used by a tier-sourced model when the task section sets no
key/base of its own.

A tier that is unset falls back to the nearest set tier (nearest *lower* first,
then higher), emitting a warning — so a minimal config that sets only, say,
``standard`` still resolves every task.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from .errors import DgmlError, ModelsConfigInvalid

# Cheapest → strongest. Fallback searches lower (cheaper) neighbours first.
TIERS: tuple[str, ...] = ("light", "standard", "advanced", "expert")

# Tier fallbacks already reported this process, so a per-file loop (e.g. bulk
# extract) doesn't flood stderr with the same line. Keyed by (requested, used).
_WARNED_TIER_FALLBACKS: set[tuple[str, str]] = set()


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed ``[models]`` block. Every field is optional; a task errors only
    when both its override and every tier are unset."""

    light: str | None = None
    standard: str | None = None
    advanced: str | None = None
    expert: str | None = None
    light_api_key_env: str | None = None
    standard_api_key_env: str | None = None
    advanced_api_key_env: str | None = None
    expert_api_key_env: str | None = None
    light_api_base: str | None = None
    standard_api_base: str | None = None
    advanced_api_base: str | None = None
    expert_api_base: str | None = None

    def resolve(self, tier: str) -> tuple[str | None, str | None, str | None]:
        """Resolve ``tier`` to ``(model, api_key_env, api_base)``.

        If ``tier`` has no model, fall back to the nearest set tier — lower
        (cheaper) neighbours first, then higher — and write a warning to stderr
        (always, independent of ``--verbose``). Returns ``(None, None, None)``
        when no tier is set at all (the caller then surfaces the appropriate
        config error)."""
        if tier not in TIERS:
            raise ValueError(f"unknown model tier {tier!r}")
        actual = self._nearest_set(tier)
        if actual is None:
            return (None, None, None)
        if actual != tier and (tier, actual) not in _WARNED_TIER_FALLBACKS:
            _WARNED_TIER_FALLBACKS.add((tier, actual))
            sys.stderr.write(
                f"[dgml] model tier '{tier}' is not set; falling back to '{actual}' "
                f"('{getattr(self, actual)}'). Set [models].{tier} to silence this.\n"
            )
        return (
            getattr(self, actual),
            getattr(self, f"{actual}_api_key_env"),
            getattr(self, f"{actual}_api_base"),
        )

    def _nearest_set(self, tier: str) -> str | None:
        idx = TIERS.index(tier)
        # Lower (cheaper) neighbours nearest-first, then higher neighbours.
        order = list(range(idx - 1, -1, -1)) + list(range(idx + 1, len(TIERS)))
        if getattr(self, tier) is not None:
            return tier
        for i in order:
            if getattr(self, TIERS[i]) is not None:
                return TIERS[i]
        return None


def resolve_task_creds(
    *,
    section_api_key: str | None,
    section_api_key_env: str | None,
    section_api_base: str | None,
    from_tier: bool,
    tier_api_key_env: str | None,
    tier_api_base: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve a task's ``(api_key, api_key_env, api_base)`` names.

    Precedence: an explicit section ``api_key``/``api_key_env``/``api_base``
    wins; otherwise, when the model came from a tier (``from_tier``), the tier's
    ``<tier>_api_key_env`` / ``<tier>_api_base`` apply; otherwise ``None`` (the
    caller/litellm falls back to the provider's conventional env var). Stores
    *names*, not concrete keys — the env-var lookup happens in each section's
    own ``resolve_*_api_key``."""
    if section_api_key is not None:
        api_key: str | None = section_api_key
        api_key_env: str | None = None
    elif section_api_key_env is not None:
        api_key, api_key_env = None, section_api_key_env
    elif from_tier and tier_api_key_env is not None:
        api_key, api_key_env = None, tier_api_key_env
    else:
        api_key, api_key_env = None, None
    api_base: str | None
    if section_api_base is not None:
        api_base = section_api_base
    else:
        api_base = tier_api_base if from_tier else None
    return api_key, api_key_env, api_base


def _validate_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelsConfigInvalid(f"'models.{field}' must be a non-empty string if set")
    return value


def load_models_config(merged: dict[str, Any]) -> ModelsConfig:
    """Build a :class:`ModelsConfig` from the merged config mapping's
    ``[models]`` section (an empty section yields an all-``None`` config)."""
    section = merged.get("models")
    if section is None:
        return ModelsConfig()
    if not isinstance(section, dict):
        raise ModelsConfigInvalid("'models' must be a table")
    fields: dict[str, str | None] = {}
    for tier in TIERS:
        fields[tier] = _validate_optional_str(section.get(tier), tier)
        for suffix in ("api_key_env", "api_base"):
            key = f"{tier}_{suffix}"
            fields[key] = _validate_optional_str(section.get(key), key)
    return ModelsConfig(**fields)


@dataclass(frozen=True)
class ResolvedModel:
    """A task's resolved model id plus its (name-only) credentials."""

    model: str
    api_key: str | None
    api_key_env: str | None
    api_base: str | None


def resolve_tiered_model(
    merged: dict[str, Any],
    *,
    section_name: str,
    tier: str,
    invalid: type[DgmlError],
    missing: type[DgmlError],
    model_field: str = "model",
    key_field: str = "api_key",
    env_field: str = "api_key_env",
    base_field: str = "api_base",
) -> ResolvedModel:
    """Resolve one task's model + credentials from the ``[{section_name}]``
    section of *merged*, or — when the section names no model — from its
    ``[models]`` *tier*.

    The section's ``model_field`` overrides the tier; ``key_field`` /
    ``env_field`` / ``base_field`` are that task's credential fields (they vary
    per task: e.g. ``label_api_key`` for generation labeling, ``schema_api_key``
    for grounded schema-gen). When the model comes from a tier and the section
    sets no credentials, the tier's ``<tier>_api_key_env`` / ``<tier>_api_base``
    apply.

    Raises ``invalid`` for a malformed value or a literal+env-name clash, and
    ``missing`` when neither the field nor the tier resolves a model. Callers
    that treat a section's mere presence as a feature switch (``style`` /
    ``text_extraction``) check that themselves before calling this.
    """
    section = merged.get(section_name)
    sec: dict[str, Any] = section if isinstance(section, dict) else {}

    def _opt_str(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise invalid(f"'{section_name}.{field}' must be a non-empty string if set")
        return value

    model = _opt_str(sec.get(model_field), model_field)
    sec_key = _opt_str(sec.get(key_field), key_field)
    sec_env = _opt_str(sec.get(env_field), env_field)
    sec_base = _opt_str(sec.get(base_field), base_field)
    if sec_key is not None and sec_env is not None:
        raise invalid(
            f"set at most one of '{section_name}.{key_field}' / "
            f"'{section_name}.{env_field}', not both"
        )

    from_tier = model is None
    tier_env: str | None = None
    tier_base: str | None = None
    if from_tier:
        model, tier_env, tier_base = load_models_config(merged).resolve(tier)
    if not isinstance(model, str) or not model.strip():
        raise missing(
            f"no {model_field} for {section_name}: set [models].{tier} or "
            f"'{section_name}.{model_field}' in the config"
        )

    api_key, api_key_env, api_base = resolve_task_creds(
        section_api_key=sec_key,
        section_api_key_env=sec_env,
        section_api_base=sec_base,
        from_tier=from_tier,
        tier_api_key_env=tier_env,
        tier_api_base=tier_base,
    )
    return ResolvedModel(model=model, api_key=api_key, api_key_env=api_key_env, api_base=api_base)
