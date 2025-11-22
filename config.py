# Copyright 2025 Cedric Sebastian
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
Configuration module for CedLM Autonomous Researcher.

Adjust these parameters to customize research behavior and quality standards.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ResearchConfig:
    """Configuration for research quality and iteration parameters."""

    # Model Configuration
    model: str = "gemini-3-pro-preview"
    temperature: float = 0.2  # Lower = more focused, factual

    # Quality Thresholds
    min_word_count: int = 500  # Minimum words per section
    min_sources: int = 3  # Minimum sources per section
    min_completeness: float = 70.0  # Minimum completeness percentage

    # Iteration Control
    max_iterations: int = 8  # Maximum research iterations
    sections_per_iteration: int = 5  # Max sections to research per iteration

    # Quality Weights
    examples_weight: float = 0.8  # % of sections that should have examples
    technical_weight: float = 0.7  # % of sections that should have technical depth

    max_thoughts: int = 32768  # Maximum thoughts budget for planner
    enable_safety: bool = True  # Enable safety checks during research

    # Output Configuration
    output_dir: str = "output"
    enable_logging: bool = True
    log_file: str = "cedlm_research.log"

    # Display Configuration
    show_progress: bool = True
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "min_word_count": self.min_word_count,
            "min_sources": self.min_sources,
            "min_completeness": self.min_completeness,
            "max_iterations": self.max_iterations,
            "sections_per_iteration": self.sections_per_iteration,
            "examples_weight": self.examples_weight,
            "technical_weight": self.technical_weight,
            "max_thoughts": self.max_thoughts,
            "enable_safety": self.enable_safety,
            "output_dir": self.output_dir,
            "enable_logging": self.enable_logging,
            "log_file": self.log_file,
            "show_progress": self.show_progress,
            "verbose": self.verbose,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ResearchConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def validate(self) -> bool:
        """Validate configuration parameters."""
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

        return True


# Default configuration instance
DEFAULT_CONFIG = ResearchConfig()


# Preset configurations for different use cases

QUICK_RESEARCH = ResearchConfig(
    min_word_count=300,
    min_sources=2,
    min_completeness=60.0,
    max_iterations=3,
    examples_weight=0.6,
    technical_weight=0.5,
)

STANDARD_RESEARCH = ResearchConfig(
    min_word_count=500,
    min_sources=3,
    min_completeness=70.0,
    max_iterations=5,
    examples_weight=0.8,
    technical_weight=0.7,
)

DEEP_RESEARCH = ResearchConfig(
    min_word_count=800,
    min_sources=5,
    min_completeness=85.0,
    max_iterations=10,
    examples_weight=0.9,
    technical_weight=0.8,
)

COMPREHENSIVE_RESEARCH = ResearchConfig(
    min_word_count=1000,
    min_sources=7,
    min_completeness=90.0,
    max_iterations=15,
    examples_weight=0.95,
    technical_weight=0.9,
)


def get_config_by_name(name: str) -> ResearchConfig:
    """
    Get a preset configuration by name.

    Args:
        name: Configuration name (quick, standard, deep, comprehensive)

    Returns:
        ResearchConfig instance

    Raises:
        ValueError: If configuration name is invalid
    """
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
            f"Invalid config name: {name}. "
            f"Valid options: {', '.join(configs.keys())}"
        )

    return configs[name_lower]


def print_config(config: ResearchConfig) -> None:
    """Print configuration in a readable format."""
    print("\n" + "="*80)
    print("⚙️  RESEARCH CONFIGURATION")
    print("="*80)
    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")
    print(f"Max Thoughts (Planner): {config.max_thoughts}")
    print(
        f"Safety Settings: {'Enabled' if config.enable_safety else 'Disabled'}")
    print(f"\n📊 Quality Standards:")
    print(f"  - Min Words/Section: {config.min_word_count}")
    print(f"  - Min Sources/Section: {config.min_sources}")
    print(f"  - Min Completeness: {config.min_completeness}%")
    print(f"  - Examples Weight: {config.examples_weight * 100}%")
    print(f"  - Technical Weight: {config.technical_weight * 100}%")
    print(f"\n🔄 Iteration Control:")
    print(f"  - Max Iterations: {config.max_iterations}")
    print(f"  - Sections/Iteration: {config.sections_per_iteration}")
    print(f"\n📁 Output:")
    print(f"  - Directory: {config.output_dir}")
    print(f"  - Log File: {config.log_file}")
    print(f"  - Logging: {'Enabled' if config.enable_logging else 'Disabled'}")
    print(f"  - Verbose: {'Yes' if config.verbose else 'No'}")
    print("="*80 + "\n")
