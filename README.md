# Learning Agent ADK: Develop Learning Material Through Command Line using Improved Deep Research Agent

**Powered by Google Agent Development Kit (ADK) & Gemini 3 Pro**

> **Truly Helpful Researcher that Iteratively Explores the Requested Topics Until Depth Understanding is Reached**

## 🎯 Overview

Learning Agent ADK is a **Improved Deep Research Agent** that performs rigorous, iterative deep research on any topic and generates comprehensive markdown curriculum through the command line. The agent **autonomously decides** when to continue researching based on quality metrics until depth understanding is achieved.

### What Makes It Truly Helpful?

- 🔄 **Dynamic Iteration**: Continues researching until quality standards are met (not fixed iterations)
- 🎯 **Section-by-Section Research**: Identifies and researches specific curriculum sections
- 📊 **Quality-Driven Decisions**: Evaluates completeness and decides autonomously to continue or proceed
- 🧠 **Self-Assessment**: Monitors depth scores, source counts, and knowledge gaps
- ⚡ **Adaptive Learning**: Identifies weak sections and targets them for improvement

## ✨ Key Features

### 1. 🧠 Autonomous Intelligence
- **Self-directed research**: Identifies knowledge gaps and explores them autonomously
- **Quality assessment**: Evaluates its own research depth after each iteration
- **Dynamic looping**: Continues until quality standards are met (up to max iterations)
- **Smart source tracking**: Catalogs and references all sources with metadata
- **Iterative refinement**: Multiple research passes with continuous improvement

### 2. 📚 Curriculum Generation
- **Structured learning paths**: Breaks topics into logical modules and subtopics
- **Depth markers**: Indicates basic (⭐), intermediate (⭐⭐), and advanced (⭐⭐⭐) sections
- **Learning objectives**: Clear goals for each module
- **Prerequisites identification**: Lists foundational knowledge needed

### 3. 🔍 Deep Research Methodology
- **Multi-source verification**: Gathers information from diverse credible sources
- **Technical precision**: Includes algorithms, architectures, and implementation details
- **Practical examples**: Real-world case studies and applications
- **Comparative analysis**: Discusses different approaches and tradeoffs

### 4. 📄 Professional Output
- **Markdown curriculum**: Complete, publication-ready documentation
- **Automatic citations**: All claims linked to sources
- **Visual enhancements**: Emojis, tables, code blocks, and callouts
- **Multiple formats**: Main curriculum, quick reference, and metadata JSON

## 🏗️ Architecture

### Autonomous Iteration Pipeline

```
User Query
    ↓
┌─────────────────────────────────────────────────────────┐
│         IterativeResearchAgent (Custom Agent)           │
│                                                          │
│  Phase 1: Curriculum Planning                          │
│    └─ Creates structured outline with sections          │
│                                                          │
│  Phase 2: Knowledge Gap Analysis                       │
│    └─ Identifies areas needing research                │
│                                                          │
│  Phase 3: Iterative Research Loop                      │
│    ┌──────────────────────────────────┐                │
│    │ WHILE quality < threshold AND    │                │
│    │       iteration < max_iterations:│                │
│    │                                  │                │
│    │  1. Identify weak sections       │                │
│    │  2. Research each section        │                │
│    │  3. Assess depth & quality       │                │
│    │  4. Evaluate overall completeness│                │
│    │  5. Refinement decision          │                │
│    │     ├─ CONTINUE → Loop back      │                │
│    │     └─ PROCEED → Exit loop       │                │
│    └──────────────────────────────────┘                │
│                                                          │
│  Phase 4: Report Synthesis                             │
│    └─ Consolidates all research with citations         │
│                                                          │
│  Phase 5: Markdown Generation                          │
│    └─ Creates final curriculum document                │
└─────────────────────────────────────────────────────────┘
    ↓
Outputs: Curriculum.md, Metadata.json, Quick Reference.md
```

### Quality Metrics & Decision Logic

The agent tracks these metrics for each section:
- **Word Count**: Aim for 500+ words per section
- **Source Count**: Minimum 3-5 credible sources
- **Examples**: Presence of practical examples
- **Technical Details**: Depth of technical explanations
- **Completeness**: Overall quality score (0-100%)

**Iteration continues if:**
- ❌ Any section < 70% completeness
- ❌ Average sources < 3 per section
- ❌ < 80% of sections have examples
- ❌ Knowledge gaps remain unfilled
- ✅ AND iteration < max_iterations

**Proceeds to synthesis when:**
- ✅ All sections ≥ 70% completeness
- ✅ Quality standards met
- ✅ OR max iterations reached

## 🚀 Setup & Usage

### Prerequisites

```bash
# Use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Python 3.10+
python --version

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Update the .env file with your Google API key
nano .env

### Configuration

Create a `.env` file with your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

### Running the Agent

**Interactive Mode (Recommended):**
```bash
cd learning_agent
python main.py
```

**Command Line Mode:**
```bash
python main.py "Your Curriculum Topic"
```

**Examples:**
```bash
# AI/ML Topics
python main.py "Retrieval-Augmented Generation in Large Language Models"
python main.py "Diffusion Models for Image Generation"

# Programming Concepts
python main.py "Advanced Rust Memory Management and Ownership"
python main.py "Functional Programming Patterns in Python"

# Scientific Topics
python main.py "CRISPR Gene Editing Techniques and Applications"
python main.py "Quantum Computing Algorithms and Implementations"

# Business/Tech
python main.py "Decentralized Finance (DeFi) Architecture"
python main.py "Microservices Architecture Best Practices"
```

### Research Configuration Presets

You can customize research quality by editing `config.py`:

```python
from config import QUICK_RESEARCH, STANDARD_RESEARCH, DEEP_RESEARCH, COMPREHENSIVE_RESEARCH

# Quick Research (3 iterations, 300 words/section)
config = QUICK_RESEARCH

# Standard Research (5 iterations, 500 words/section) - Default
config = STANDARD_RESEARCH

# Deep Research (10 iterations, 800 words/section)
config = DEEP_RESEARCH

# Comprehensive Research (15 iterations, 1000 words/section)
config = COMPREHENSIVE_RESEARCH
```

Or select a preset at runtime without editing code by setting an environment variable:

```bash
export CEDLM_RESEARCH_PRESET=deep   # quick | standard | deep | comprehensive | default
python main.py "Your Research Topic"
```

**Preset Comparison:**

| Preset | Min Words | Min Sources | Min Complete | Max Iterations |
|--------|-----------|-------------|--------------|----------------|
| Quick | 300 | 2 | 60% | 3 |
| Standard | 500 | 3 | 70% | 5 |
| Deep | 800 | 5 | 85% | 10 |
| Comprehensive | 1000 | 7 | 90% | 15 |

### Output Files

The agent generates three files in the `output/` directory:

1. **`{topic}_{timestamp}.md`** - Complete curriculum (10-50 pages)
   - Title and metadata
   - Table of contents
   - Curriculum overview
   - Detailed research report with citations
   - Research metrics table
   - Full bibliography

2. **`{topic}_metadata_{timestamp}.json`** - Research metadata
   - Iteration count
   - Source list with URLs
   - Depth scores per section
   - Knowledge gaps identified
   - Explored topics

3. **`{topic}_quick_reference.md`** - Quick reference guide
   - Curriculum structure outline
   - Top 10 resources
   - Study tips

## 📊 Example Research Session

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    Learning Agent v2.0                           ║
║           Rigorous Iterative Research Until Deep Understanding                ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📋 Topic: Machine Learning Transformers

🚀LEARNING AGENT
📋 Research Topic: Machine Learning Transformers
⏰ Started: 2025-11-20 14:30:00

🔄 Initiating autonomous research process...

📚 Phase 1: Creating Curriculum Structure...
✅ Curriculum created with 12 sections

🔍 Phase 2: Analyzing Knowledge Gaps...
✅ Knowledge gaps identified

🔄 Phase 3: Iterative Deep Research
Target: 500+ words, 3+ sources, 70%+ completeness per section
Max iterations: 8

🔄 Iteration 1/8
   Researching 5 sections...
   [1/5] Researching: Introduction to Transformers...
      ⚠️ 45.2% complete (226 words)
   [2/5] Researching: Attention Mechanisms...
      ⚠️ 58.3% complete (291 words)
   ...
   
   📊 Evaluating research quality...
   Overall Completeness: 52.4%
   Complete Sections: 2/12
   
   🤔 Assessing if more research is needed...
   ➡️  Continuing research (Quality: 52.4%)

🔄 Iteration 2/8
   Researching 5 sections...
   [1/5] Researching: Introduction to Transformers...
      ✅ 78.5% complete (392 words)
   ...

   📊 Evaluating research quality...
   Overall Completeness: 76.8%
   Complete Sections: 10/12
   
   🤔 Assessing if more research is needed...
   ✅ Research quality standards met! Proceeding to synthesis.

✨ Phase 4: Synthesizing Research Report...
✅ Report synthesized with citations

📝 Phase 5: Generating Markdown Curriculum...
✅ Final curriculum generated

================================================================================
🎉 AUTONOMOUS RESEARCH COMPLETE!
================================================================================
✅ Iterations: 2
✅ Sources: 37
✅ Overall Quality: 76.8%
✅ Sections: 12
================================================================================

📚 RESEARCH SUMMARY
🔄 Iterations Completed: 2
📖 Total Sources Referenced: 37
🎯 Topics Explored: 12
⚠️  Knowledge Gaps Identified: 8

✅ Curriculum saved to: output/Machine_Learning_Transformers_20251120_143000.md
📊 Metadata: output/Machine_Learning_Transformers_metadata_20251120_143000.json
⚡ Quick reference: output/Machine_Learning_Transformers_quick_reference.md
```

## 🔧 Advanced Configuration

### Customize Research Parameters

Edit `learning_agent/config.py`:

```python
from config import ResearchConfig

# Create custom configuration
custom_config = ResearchConfig(
    model="gemini-3-pro-preview",
    temperature=0.2,
    min_word_count=600,        # More detailed sections
    min_sources=4,              # More sources required
    min_completeness=80.0,      # Higher quality threshold
    max_iterations=12,          # More iterations allowed
    sections_per_iteration=3,   # Focus on fewer sections
)

# Validate configuration
custom_config.validate()
```

### Modify Quality Thresholds

```python
# In agent.py, adjust these constants:
MIN_WORD_COUNT = 500      # Minimum words per section
MIN_SOURCES = 3           # Minimum sources per section
MIN_COMPLETENESS = 70.0   # Minimum completeness percentage
MAX_ITERATIONS = 8        # Maximum research iterations
```

## 🎓 Use Cases

### Educational
- 📚 Create comprehensive course materials
- 📝 Generate study guides for exams
- 🎯 Build learning pathways for new topics
- 🏫 Curriculum development

### Professional
- 💼 Research emerging technologies
- 📊 Competitive analysis and market research
- 📈 Industry trend reports
- 🔬 Technical documentation

### Academic
- 📖 Literature review assistance
- 🔍 Research topic exploration
- 📚 Bibliography generation
- 🎓 Knowledge synthesis

### Personal
- 🚀 Self-directed learning
- 💡 Skill development paths
- 📚 Knowledge base creation
- 🧠 Deep understanding of complex topics

## 🔬 Technical Details

### Built With
- **Google ADK**: Agent orchestration and state management
- **Gemini 3 Pro**: Advanced reasoning and research
- **BuiltInPlanner**: Extended thinking for complex analysis
- **Google Search Tool**: Multi-source information gathering

### Agent Architecture
- **IterativeResearchAgent**: Custom agent class with while-loop control
- **LLmAgent**: Individual specialized research agents
- **CallbackContext**: State management and quality tracking
- **Dynamic Prompting**: Context-aware research queries

### Key Innovations
1. **Dynamic Iteration**: True while-loop with quality-based exit conditions
2. **Section Tracking**: Maps curriculum sections to research content
3. **Quality Evaluation**: Multi-dimensional scoring system
4. **Autonomous Decision**: Agent decides when research is complete
5. **Gap Identification**: Automatically finds and fills knowledge gaps

## 📝 Logging

All research activities are logged to `cedlm_research.log`:
- Research queries and responses
- Source discoveries and tracking
- Quality assessments per iteration
- Decision-making rationale
- Error tracking and debugging

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional research tools (web scraping, PDF parsing, arXiv search)
- More sophisticated quality metrics
- Multi-language support
- Custom output formats (PDF, HTML, LaTeX)
- Integration with knowledge bases (Notion, Obsidian)
- Real-time progress streaming

## 📄 License

Licensed under the Apache License, Version 2.0

## 🙏 Acknowledgments

- Google Agent Development Kit (ADK)
- Google Gemini 3 Pro
- The open-source AI community

---

**Made with ❤️ by Cedric Sebastian**
