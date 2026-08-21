# Copyright 2026 Cedric Sebastian
#
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

"""Kythe: Agentic Deep Research and Curriculum Learning Agent."""

import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

try:
    from . import agent, config, exporters, models, orchestrator, persistence, quality, reader, tools
    from .config import ResearchConfig
    from .models import CurriculumModel, CurriculumOutline, OutlineSection, Run, Section
    from .orchestrator import RunOrchestrator
    from .persistence import PersistenceRepository
    from .quality import DeterministicQualityEvaluator
    from .reader import DeepReader
except (ImportError, ValueError):
    import agent, config, exporters, models, orchestrator, persistence, quality, reader, tools
    from config import ResearchConfig
    from models import CurriculumModel, CurriculumOutline, OutlineSection, Run, Section
    from orchestrator import RunOrchestrator
    from persistence import PersistenceRepository
    from quality import DeterministicQualityEvaluator
    from reader import DeepReader

__all__ = [
    "agent",
    "config",
    "models",
    "persistence",
    "reader",
    "quality",
    "orchestrator",
    "exporters",
    "tools",
    "ResearchConfig",
    "RunOrchestrator",
    "PersistenceRepository",
    "DeepReader",
    "DeterministicQualityEvaluator",
    "CurriculumModel",
    "CurriculumOutline",
    "OutlineSection",
    "Run",
    "Section",
]
