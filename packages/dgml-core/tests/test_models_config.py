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

"""The ``[models]`` tier block: resolution, fallback, and the stderr warning."""

from __future__ import annotations

import pytest
from dgml_core import models_config
from dgml_core.models_config import ModelsConfig


@pytest.fixture(autouse=True)
def _clear_fallback_dedup() -> None:
    """The fallback warning is deduped per process; reset it so each test starts
    from a clean slate."""
    models_config._WARNED_TIER_FALLBACKS.clear()


def test_resolve_exact_tier_no_warning(capsys: pytest.CaptureFixture[str]) -> None:
    m = ModelsConfig(light="a", standard="b", advanced="c", expert="d")
    assert m.resolve("advanced") == ("c", None, None)
    assert capsys.readouterr().err == ""


def test_resolve_carries_tier_credentials(capsys: pytest.CaptureFixture[str]) -> None:
    m = ModelsConfig(advanced="c", advanced_api_key_env="MY_KEY", advanced_api_base="http://x")
    assert m.resolve("advanced") == ("c", "MY_KEY", "http://x")
    assert capsys.readouterr().err == ""


def test_resolve_prefers_nearest_lower_tier(capsys: pytest.CaptureFixture[str]) -> None:
    # expert unset; both standard and light set → nearest lower is standard.
    m = ModelsConfig(light="l", standard="s")
    model, _, _ = m.resolve("expert")
    assert model == "s"
    assert "falling back to 'standard'" in capsys.readouterr().err


def test_resolve_falls_back_upward_when_nothing_below(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Only standard set; light has no lower neighbour → nearest higher is standard.
    m = ModelsConfig(standard="only-standard")
    model, _, _ = m.resolve("light")
    assert model == "only-standard"
    err = capsys.readouterr().err
    assert "tier 'light' is not set" in err
    assert "falling back to 'standard'" in err


def test_fallback_warning_is_deduped(capsys: pytest.CaptureFixture[str]) -> None:
    m = ModelsConfig(standard="only-standard")
    m.resolve("light")
    first = capsys.readouterr().err
    m.resolve("light")
    second = capsys.readouterr().err
    assert first.count("falling back") == 1
    assert second == ""  # same (tier, used) pair — not repeated


def test_resolve_none_when_no_tier_set(capsys: pytest.CaptureFixture[str]) -> None:
    assert ModelsConfig().resolve("standard") == (None, None, None)
    assert capsys.readouterr().err == ""  # nothing to fall back to → no warning
