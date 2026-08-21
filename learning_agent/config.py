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

"""
Configuration module for Kythe Autonomous Deep Research Agent.

Defines research quality parameters, deep reader constraints, deterministic
gates, antislop filtering rules, and export specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


DEFAULT_BANNED_SLOP_WORDS: List[str] = [
    "delve",
    "delving",
    "testament",
    "tapestry",
    "landscape",
    "pivotal",
    "fostering",
    "crucial",
    "vibrant",
    "beacon",
    "groundbreaking",
    "game-changer",
    "revolutionize",
    "unwavering",
    "undeniably",
    "comprehensive exploration",
    "serves as a testament",
    "stands as a",
    "in today's rapidly evolving",
    "nestled",
    "bustling",
]


@dataclass
class ResearchConfig:
    """Configuration for research quality, deep reading, and exports."""

    # Model Configuration
    model: str = "gemini-3-pro-preview"
    temperature: float = 0.2  # Low temperature for factual precision

    # Quality Thresholds
    min_word_count: int = 500  # Minimum words per section
    min_sources: int = 3  # Minimum sources per section
    min_completeness: float = 70.0  # Minimum completeness percentage

    # Iteration and Batch Control
    max_iterations: int = 5  # Maximum research passes per section
    sections_per_iteration: int = 5  # Max distinct sections in a batch

    # Quality Weights and Evidence
    examples_weight: float = 0.8  # Target fraction of sections with code/examples
    technical_weight: float = 0.7  # Target fraction of sections with technical depth
    min_citation_coverage: float = 0.6  # Minimum ratio of cited substantive units (0.0 to 1.0)
    min_unique_domains: int = 2  # Minimum distinct source domains per section

    # Antislop Enforcement
    enable_antislop: bool = True  # Enforce antislop vocabulary and punctuation filters
    banned_words: List[str] = field(default_factory=lambda: list(DEFAULT_BANNED_SLOP_WORDS))

    # Deep Reader and Network Limits
    max_queries_per_pass: int = 3  # Search queries generated per pass
    max_candidates_per_query: int = 5  # Candidate URLs considered per query
    max_pages_per_pass: int = 4  # Full-page fetches per pass
    fetch_timeout_s: float = 10.0  # Network timeout in seconds per fetch
    max_page_bytes: int = 2_000_000  # Maximum body size (2MB)
    cache_ttl_hours: int = 72  # Cache lifetime for fetched pages

    # Thought Budget and Safety
    max_thoughts: int = 32768  # Thoughts budget for reasoning stages
    enable_safety: bool = True  # Safety filter enablement

    # Persistence and Workflow
    database_path: str = "research.db"  # SQLite database path
    require_outline_approval: bool = True  # Require interactive review of outline
    allow_exhausted_sections: bool = True  # Allow final export with exhausted sections

    # Output and Export Configuration
    output_dir: str = "output"
    export_formats: List[str] = field(default_factory=lambda: [
        "markdown", "html", "pdf", "quiz", "flashcards", "metadata"
    ])
    enable_logging: bool = True
    log_file: str = "kythe_research.log"

    # Display Configuration
    show_progress: bool = True
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ResearchConfig":
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def validate(self) -> bool:
        """Validate configuration parameters and constraints."""
        if self.min_word_count < 100:
            raise ValueError("min_word_count must be at least 100")

        if self.min_sources < 1:
            raise ValueError("min_sources must be at least 1")

        if not (0 <= self.min_completeness <= 100):
            raise ValueError("min_completeness must be between 0 and 100")

        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")

        if self.max_thoughts < 1:
            raise ValueError("max_thoughts must be at least 1")

        if self.max_pages_per_pass < 1:
            raise ValueError("max_pages_per_pass must be at least 1")

        if self.fetch_timeout_s <= 0:
            raise ValueError("fetch_timeout_s must be positive")

        if self.max_page_bytes <= 1024:
            raise ValueError("max_page_bytes must be at least 1024 bytes")

        if not (0.0 <= self.min_citation_coverage <= 1.0):
            raise ValueError("min_citation_coverage must be between 0.0 and 1.0")

        if self.min_unique_domains < 1:
            raise ValueError("min_unique_domains must be at least 1")

        if self.max_queries_per_pass < 1:
            raise ValueError("max_queries_per_pass must be at least 1")

        if self.max_candidates_per_query < 1:
            raise ValueError("max_candidates_per_query must be at least 1")

        return True


# Default configuration instance
DEFAULT_CONFIG = ResearchConfig()
DB_PATH = DEFAULT_CONFIG.database_path

# Preset configurations
QUICK_RESEARCH = ResearchConfig(
    min_word_count=300,
    min_sources=2,
    min_completeness=60.0,
    max_iterations=2,
    examples_weight=0.6,
    technical_weight=0.5,
    max_pages_per_pass=3,
    max_queries_per_pass=2,
    min_citation_coverage=0.5,
    min_unique_domains=1,
)

STANDARD_RESEARCH = ResearchConfig(
    min_word_count=500,
    min_sources=3,
    min_completeness=70.0,
    max_iterations=4,
    examples_weight=0.8,
    technical_weight=0.7,
    max_pages_per_pass=4,
    max_queries_per_pass=3,
    min_citation_coverage=0.6,
    min_unique_domains=2,
)

DEEP_RESEARCH = ResearchConfig(
    min_word_count=800,
    min_sources=5,
    min_completeness=85.0,
    max_iterations=6,
    examples_weight=0.9,
    technical_weight=0.8,
    max_pages_per_pass=6,
    max_queries_per_pass=4,
    min_citation_coverage=0.7,
    min_unique_domains=3,
)

COMPREHENSIVE_RESEARCH = ResearchConfig(
    min_word_count=1000,
    min_sources=7,
    min_completeness=90.0,
    max_iterations=8,
    examples_weight=0.95,
    technical_weight=0.9,
    max_pages_per_pass=8,
    max_queries_per_pass=5,
    min_citation_coverage=0.8,
    min_unique_domains=4,
)


def list_preset_names() -> List[str]:
    """Return available configuration preset names."""
    return ["quick", "standard", "deep", "comprehensive", "default"]


def get_config_by_name(name: str) -> ResearchConfig:
    """Get preset configuration by name."""
    configs = {
        "quick": QUICK_RESEARCH,
        "standard": STANDARD_RESEARCH,
        "deep": DEEP_RESEARCH,
        "comprehensive": COMPREHENSIVE_RESEARCH,
        "default": DEFAULT_CONFIG,
    }

    name_lower = name.lower()
    if name_lower not in configs:
        raise ValueError(
            f"Invalid config name: '{name}'. "
            f"Valid options: {', '.join(configs.keys())}"
        )

    return configs[name_lower]


def print_config(config: ResearchConfig) -> None:
    """Print configuration in clean terminal format without emojis."""
    print("\n" + "=" * 80)
    print("KYTHE RESEARCH CONFIGURATION")
    print("=" * 80)
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Database: {config.database_path}")
    print(f"Max Thoughts (Planner): {config.max_thoughts}")
    print(f"Safety Settings: {'Enabled' if config.enable_safety else 'Disabled'}")
    print(f"Require Outline Approval: {'Yes' if config.require_outline_approval else 'No (Auto-Approve)'}")
    print(f"Antislop Filtering: {'Enabled' if config.enable_antislop else 'Disabled'}")
    print("\nQuality Standards:")
    print(f"  - Min Words/Section: {config.min_word_count}")
    print(f"  - Min Sources/Section: {config.min_sources}")
    print(f"  - Min Unique Domains/Section: {config.min_unique_domains}")
    print(f"  - Min Completeness: {config.min_completeness}%")
    print(f"  - Min Citation Coverage: {config.min_citation_coverage * 100:.0f}%")
    print(f"  - Examples Weight: {config.examples_weight * 100:.0f}%")
    print(f"  - Technical Weight: {config.technical_weight * 100:.0f}%")
    print("\nIteration & Reader Limits:")
    print(f"  - Max Passes/Section: {config.max_iterations}")
    print(f"  - Sections/Batch: {config.sections_per_iteration}")
    print(f"  - Max Queries/Pass: {config.max_queries_per_pass}")
    print(f"  - Max Pages Fetched/Pass: {config.max_pages_per_pass}")
    print(f"  - Fetch Timeout: {config.fetch_timeout_s}s")
    print(f"  - Max Page Size: {config.max_page_bytes / 1024:.0f} KB")
    print("\nOutput & Exports:")
    print(f"  - Directory: {config.output_dir}")
    print(f"  - Formats: {', '.join(config.export_formats)}")
    print(f"  - Log File: {config.log_file}")
    print("=" * 80 + "\n")
