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

"""Default `[models]` tables `dgml init --provider` writes into a user config.

Kept apart from :mod:`dgml_core.storage` (the path-resolution and TOML-writing
machinery) so the shipped model defaults live in one obvious place and can be
updated without touching the config generator.
"""

from __future__ import annotations

# The four tiers, cheapest → strongest, per provider. `dgml init --provider`
# writes one of these into the `[models]` block; the tier→task mapping and
# per-task overrides are documented in the CLI reference, not baked into the
# file (the mapping may change without a config rewrite).
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "mixed": {
        # Gemini Flash-Lite for the cheap high-volume vision work
        # (classification/style); Anthropic for the document-reasoning pipeline
        # (transcription → labeling/value-extraction → schema generation).
        "light": "gemini/gemini-2.5-flash-lite",
        "standard": "anthropic/claude-haiku-4-5",
        "advanced": "anthropic/claude-sonnet-4-6",
        "expert": "anthropic/claude-opus-4-8",
    },
    "anthropic": {
        "light": "anthropic/claude-haiku-4-5",
        "standard": "anthropic/claude-haiku-4-5",
        "advanced": "anthropic/claude-sonnet-4-6",
        "expert": "anthropic/claude-opus-4-8",
    },
    "google": {
        "light": "gemini/gemini-2.5-flash-lite",
        "standard": "gemini/gemini-2.5-flash",
        "advanced": "gemini/gemini-2.5-pro",
        "expert": "gemini/gemini-2.5-pro",
    },
}
