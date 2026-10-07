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

"""`extract_file` — the library's `dgml extraction extract`.

Two contracts: a file must be assigned to the docset before it can be extracted
into it (the artifact lives under the pair's prefix, which only the assignment
makes reachable), and the call resolves its own config (overriding only the
values model/effort) and defaults stats off.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import (
    DocSetStore,
    FileNotFound,
    GroundedConfigInvalid,
    SchemaNotFound,
    Workspace,
    extract_file,
)
from dgml_core.grounded import ExtractionResult, GroundedConfig

_RNC = """\
namespace docset = "http://dgml.io/test/T"

Title =
  element docset:Title {
    text
  }
"""

_CONFIG = GroundedConfig(schema_model="test/schema-model", values_model="test/values-model")


def _docset(workspace: Workspace, *, schema: bool = True) -> str:
    store = DocSetStore(workspace)
    ds = store.create(name="T")
    if schema:
        store.set_schema(ds.id, _RNC)
    return ds.id


def _use_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dgml_core.grounded.load_grounded_config", lambda ws: _CONFIG)


def _capture_extract(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_extract(ws: Workspace, d: str, f: str, **kw: Any) -> ExtractionResult:
        seen.update(kw)
        return ExtractionResult(
            values={}, tool_calls=0, xml_key="k", mode="extraction", model=kw["config"].values_model
        )

    monkeypatch.setattr("dgml_core.grounded.extract_values", fake_extract)
    return seen


def test_unassigned_file_is_refused_before_any_llm_call(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    with patch("litellm.completion") as completion:
        with pytest.raises(FileNotFound, match="not assigned to docset"):
            extract_file(workspace, ds_id, "fileaaaaaaaa")
    completion.assert_not_called()


def test_no_schema_is_refused_first(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_id = _docset(workspace, schema=False)
    _use_config(monkeypatch)
    with pytest.raises(SchemaNotFound):
        extract_file(workspace, ds_id, "fileaaaaaaaa")


def test_uses_workspace_config_and_defaults_stats_off(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    seen = _capture_extract(monkeypatch)

    result = extract_file(workspace, ds_id, "fileaaaaaaaa")

    assert seen == {"config": _CONFIG, "write_stats": False, "debug": False}
    assert result.model == "test/values-model"


def test_model_and_effort_overrides(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    seen = _capture_extract(monkeypatch)

    result = extract_file(
        workspace, ds_id, "fileaaaaaaaa", values_model="test/other", values_effort="low"
    )
    assert seen["config"].values_model == "test/other"
    assert seen["config"].values_reasoning_effort == "low"
    assert result.model == "test/other"

    extract_file(workspace, ds_id, "fileaaaaaaaa", values_effort="default")
    assert seen["config"].values_reasoning_effort is None


def test_invalid_effort_is_refused(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    with pytest.raises(GroundedConfigInvalid, match="values_effort"):
        extract_file(workspace, ds_id, "fileaaaaaaaa", values_effort="turbo")
