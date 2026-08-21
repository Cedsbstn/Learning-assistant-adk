import pytest
from config import ResearchConfig
from models import GapCategory, Section, SectionStatus, Source
from quality import QualityEvaluator, calculate_citation_coverage, extract_cited_source_ids


def test_extract_cited_source_ids():
    text = """
    Rust ownership model ensures memory safety without GC <cite source="src-1"/>.
    Borrow checker verifies lifetimes at compile time <cite source='src-2' />.
    Also according to src-3, references cannot outlive referents.
    """
    ids = extract_cited_source_ids(text)
    assert "src-1" in ids
    assert "src-2" in ids
    assert "src-3" in ids


def test_calculate_citation_coverage():
    sources = {"src-1", "src-2"}
    text = (
        "Paragraph 1 with factual claims and proof <cite source=\"src-1\"/>.\n\n"
        "Paragraph 2 with more architectural insights and citations <cite source=\"src-2\"/>.\n\n"
        "Paragraph 3 with uncited statements that require backing evidence."
    )
    coverage, cited, total = calculate_citation_coverage(text, sources)
    assert total == 3
    assert cited == 2
    assert pytest.approx(coverage, 0.01) == 0.667


def test_quality_evaluator_all_pass():
    cfg = ResearchConfig(
        min_word_count=100,
        min_sources=2,
        min_unique_domains=2,
        min_citation_coverage=0.5,
    )
    evaluator = QualityEvaluator(cfg)

    sec = Section(
        section_id="sec-1",
        run_id="run-1",
        ordinal=1,
        title="Ownership & Lifetimes",
        depth_target="intermediate",
    )

    sources = [
        Source(source_id="src-1", canonical_url="https://doc.rust-lang.org/book", title="Rust Book", domain="doc.rust-lang.org"),
        Source(source_id="src-2", canonical_url="https://blog.rust-lang.org/ownership", title="Rust Blog", domain="blog.rust-lang.org"),
    ]

    draft = """
    # Ownership and Lifetimes in Rust

    Rust uses an ownership system with strict compile-time rules to manage memory safely <cite source="src-1"/>.
    Every value has a single owner variable, and when the owner goes out of scope, the memory is deallocated.

    ## Borrowing Mechanism and Technical Architecture

    Instead of transferring ownership, code can create references to data using borrowing mechanisms <cite source="src-2"/>.
    The borrow checker enforces that you may have any number of immutable references or exactly one mutable reference.

    ## Practical Example

    Here is a practical code example illustrating ownership move semantics:

    ```rust
    fn main() {
        let s1 = String::from("hello");
        let s2 = s1; // s1 moved to s2
        println!("{}", s2);
    }
    ```

    This methodology eliminates data races and use-after-free bugs entirely.
    """

    res = evaluator.evaluate_section(sec, draft, sources)
    assert res.passed is True
    assert res.score >= 80.0
    assert res.word_count >= 100
    assert res.source_count == 2
    assert res.unique_domains == 2
    assert res.has_examples is True
    assert res.has_technical_details is True
    assert len(res.open_gaps) == 0


def test_quality_evaluator_fails_and_generates_gaps():
    cfg = ResearchConfig(
        min_word_count=500,
        min_sources=3,
        min_unique_domains=2,
        min_citation_coverage=0.6,
    )
    evaluator = QualityEvaluator(cfg)

    sec = Section(
        section_id="sec-2",
        run_id="run-1",
        ordinal=2,
        title="Concurrency",
        depth_target="intermediate",
    )

    # Only 1 source from 1 domain
    sources = [
        Source(source_id="src-1", canonical_url="https://example.com/threads", title="Threads", domain="example.com"),
    ]

    # Short draft without examples, technical terms, or sufficient words
    draft = "Short draft about concurrency. Threads can run in parallel."

    res = evaluator.evaluate_section(sec, draft, sources)
    assert res.passed is False
    assert len(res.open_gaps) >= 3

    categories = {g.category for g in res.open_gaps}
    assert GapCategory.DETAIL.value in categories
    assert GapCategory.SOURCES.value in categories


def test_depth_adjusted_word_count():
    cfg = ResearchConfig(min_word_count=300, min_sources=1, min_citation_coverage=0.0)
    evaluator = QualityEvaluator(cfg)

    # Basic requires ~210 words (0.7 * 300)
    basic_sec = Section(section_id="s-b", run_id="r-1", ordinal=1, title="Basic Intro", depth_target="basic")
    basic_text = " ".join(["word"] * 220) + " example algorithm"
    res_b = evaluator.evaluate_section(basic_sec, basic_text, [Source(source_id="s-1", canonical_url="http://a.com", title="A", domain="a.com")])
    assert res_b.passed is True

    # Advanced requires ~390 words (1.3 * 300)
    adv_sec = Section(section_id="s-a", run_id="r-1", ordinal=2, title="Advanced Topic", depth_target="advanced")
    res_a = evaluator.evaluate_section(adv_sec, basic_text, [Source(source_id="s-1", canonical_url="http://a.com", title="A", domain="a.com")])
    assert res_a.passed is False  # 220 words is not enough for 390 target


def test_detect_slop_patterns():
    from quality import detect_slop_patterns
    
    slop_text = "This serves as a testament to the vibrant tapestry of the landscape \u2014 fostering innovation."
    found = detect_slop_patterns(slop_text)
    assert len(found) >= 3
    assert any("testament" in f for f in found)
    assert any("tapestry" in f for f in found)
    assert any("dash" in f for f in found)

    clean_text = "The cache implementation uses an LRU eviction policy with O(1) lookups."
    assert len(detect_slop_patterns(clean_text)) == 0


def test_quality_evaluator_catches_slop():
    cfg = ResearchConfig(
        min_word_count=100,
        min_sources=1,
        min_unique_domains=1,
        min_citation_coverage=0.5,
        enable_antislop=True,
    )
    evaluator = QualityEvaluator(cfg)

    sec = Section(
        section_id="sec-slop",
        run_id="run-1",
        ordinal=1,
        title="Modern AI Architecture",
        depth_target="intermediate",
    )
    sources = [
        Source(source_id="src-1", canonical_url="https://example.com/ai", title="AI Paper", domain="example.com")
    ]

    slop_draft = """
    # Modern AI Architecture

    Let us delve into this pivotal landscape which serves as a testament to innovation <cite source="src-1"/>.
    The vibrant tapestry of neural models — fostering next-generation capabilities — is crucial.

    ```python
    def run():
        pass
    ```
    """
    res = evaluator.evaluate_section(sec, slop_draft, sources)
    # Gaps should include style violation
    categories = {g.category.value if isinstance(g.category, GapCategory) else g.category for g in res.open_gaps}
    assert GapCategory.STYLE.value in categories
    assert res.score < 100.0
