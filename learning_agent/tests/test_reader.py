import pytest
from unittest.mock import MagicMock, patch
from models import ExtractedDocument, FetchResult
from reader import (
    canonicalize_url,
    is_safe_ip,
    validate_url_safety,
    PageExtractor,
    EvidenceChunker,
    SafeFetcher,
    DeepReader,
)
from persistence import DatabaseManager, PersistenceRepository


def test_canonicalize_url():
    raw = "HTTPS://Example.COM:443/docs/intro/?utm_source=twitter&utm_medium=social&b_param=2&a_param=1#top-heading"
    canonical = canonicalize_url(raw)
    assert canonical == "https://example.com/docs/intro/?a_param=1&b_param=2"

    raw_fb = "http://example.org/article?fbclid=IwAR123&gclid=xyz&topic=rust"
    canonical_fb = canonicalize_url(raw_fb)
    assert canonical_fb == "http://example.org/article?topic=rust"


def test_is_safe_ip():
    # Loopback
    assert is_safe_ip("127.0.0.1") is False
    assert is_safe_ip("127.0.1.10") is False
    assert is_safe_ip("::1") is False

    # Private
    assert is_safe_ip("10.0.0.1") is False
    assert is_safe_ip("172.16.0.1") is False
    assert is_safe_ip("192.168.1.1") is False
    assert is_safe_ip("fc00::1") is False

    # Link-local / Metadata
    assert is_safe_ip("169.254.169.254") is False
    assert is_safe_ip("fe80::1") is False

    # Public safe IPs
    assert is_safe_ip("8.8.8.8") is True
    assert is_safe_ip("1.1.1.1") is True
    assert is_safe_ip("93.184.216.34") is True


def test_validate_url_safety():
    # Disallowed schemes
    safe, reason = validate_url_safety("file:///C:/Windows/System32/drivers/etc/hosts")
    assert safe is False
    assert "scheme" in reason.lower()

    safe, reason = validate_url_safety("ftp://ftp.example.com/file.zip")
    assert safe is False

    # Blocked hostnames
    safe, reason = validate_url_safety("http://169.254.169.254/latest/meta-data/")
    assert safe is False

    safe, reason = validate_url_safety("http://metadata.google.internal/computeMetadata/v1/")
    assert safe is False

    safe, reason = validate_url_safety("http://localhost:8080/secret")
    assert safe is False


def test_page_extractor_html():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rust Memory Safety Guide</title>
        <style>.ads { display: block; }</style>
        <script>alert('pwn');</script>
    </head>
    <body>
        <nav><a href="/home">Home</a></nav>
        <h1>Rust Memory Management</h1>
        <p>Rust achieves memory safety without a garbage collector through its ownership system.</p>
        <h2>Borrowing & Lifetimes</h2>
        <p>References must always be valid.</p>
        <pre><code class="language-rust">
        fn main() {
            let s = String::from("hello");
            println!("{}", s);
        }
        </code></pre>
        <ul>
            <li>Each value in Rust has an owner.</li>
            <li>There can only be one owner at a time.</li>
        </ul>
        <blockquote>Rust prevents data races at compile time.</blockquote>
        <footer>Copyright 2026</footer>
    </body>
    </html>
    """
    fetch_res = FetchResult(
        requested_url="https://example.com/rust",
        final_url="https://example.com/rust",
        status_code=200,
        content_type="text/html; charset=utf-8",
        body_bytes=html.encode("utf-8"),
    )

    doc = PageExtractor.extract(fetch_res, "https://example.com/rust")
    assert doc.title == "Rust Memory Safety Guide"
    assert "Rust Memory Management" in doc.headings
    assert "Borrowing & Lifetimes" in doc.headings
    assert len(doc.code_blocks) == 1
    assert "fn main()" in doc.code_blocks[0]
    assert "alert('pwn')" not in doc.text
    assert "Copyright 2026" not in doc.text
    assert "Each value in Rust has an owner" in doc.text
    assert "Rust prevents data races" in doc.text
    assert doc.word_count > 20


def test_page_extractor_plain_text():
    text = "Simple plain text documentation for an API."
    fetch_res = FetchResult(
        requested_url="https://example.com/api.txt",
        final_url="https://example.com/api.txt",
        status_code=200,
        content_type="text/plain",
        body_bytes=text.encode("utf-8"),
    )
    doc = PageExtractor.extract(fetch_res, "https://example.com/api.txt")
    assert doc.text == text
    assert doc.extraction_method == "plain_text"
    assert doc.word_count == 7


def test_evidence_chunker():
    doc = ExtractedDocument(
        canonical_url="https://example.com/test",
        title="Test Page",
        text="Paragraph 1 with interesting facts.\n\nParagraph 2 with more architectural details.\n\nParagraph 3 with conclusion.",
        content_hash="dummyhash",
        word_count=15,
    )
    chunks = EvidenceChunker.chunk(doc, source_id="src-1", max_chunk_chars=50)
    assert len(chunks) >= 2
    assert chunks[0].source_id == "src-1"
    assert "Paragraph 1" in chunks[0].text


def test_deep_reader_cache_and_pipeline(tmp_path):
    db_file = tmp_path / "test_reader.db"
    mgr = DatabaseManager(str(db_file))
    repo = PersistenceRepository(mgr)
    reader = DeepReader(repo=repo, fetch_timeout_s=5.0)

    # Mock fetcher
    mock_html = "<html><head><title>Mocked Architecture</title></head><body><h1>System Overview</h1><p>Deep details on distributed systems.</p></body></html>"
    with patch.object(reader.fetcher, "fetch") as mock_fetch:
        mock_fetch.return_value = FetchResult(
            requested_url="https://example.com/sys",
            final_url="https://example.com/sys",
            status_code=200,
            content_type="text/html",
            body_bytes=mock_html.encode("utf-8"),
        )

        # 1st call fetches and caches
        source, chunks = reader.process_url("https://example.com/sys?utm_medium=email", source_id="src-1")
        assert source.source_id == "src-1"
        assert source.title == "Mocked Architecture"
        assert len(chunks) >= 1
        assert mock_fetch.call_count == 1

        # 2nd call should hit cache without calling fetcher again
        source2, chunks2 = reader.process_url("https://example.com/sys?utm_source=news", source_id="src-2")
        assert source2.title == "Mocked Architecture"
        assert source2.extraction_status == "cached"
        assert mock_fetch.call_count == 1
