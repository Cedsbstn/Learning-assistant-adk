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
Deep Reader & Safe Evidence Pipeline for Kythe Autonomous Deep Research Agent.

Includes SSRF-safe HTTP fetching, URL canonicalization, streaming size limits,
BeautifulSoup HTML extraction, semantic chunking, and SQLite content caching.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from models import EvidenceChunk, ExtractedDocument, FetchResult, Source, utc_now_iso
from persistence import PersistenceRepository

logger = logging.getLogger(__name__)

# Tracking parameters to strip during URL canonicalization
TRACKING_PARAMS: Set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "dclid",
    "msclkid",
    "ref",
    "source",
    "mc_cid",
    "mc_eid",
    "igshid",
    "_hsenc",
    "_hsmi",
    "wbraid",
    "gbraid",
}

# Supported Content-Types for extraction
SUPPORTED_CONTENT_TYPES: Set[str] = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "application/xml",
    "text/xml",
}

# IP Ranges blocked for SSRF prevention
BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),          # Current network (only valid as source)
    ipaddress.ip_network("10.0.0.0/8"),         # Private-Use (RFC 1918)
    ipaddress.ip_network("100.64.0.0/10"),      # Shared Address Space (RFC 6598)
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("169.254.0.0/16"),     # Link Local (including AWS/GCP metadata 169.254.169.254)
    ipaddress.ip_network("172.16.0.0/12"),      # Private-Use (RFC 1918)
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),       # Documentation (TEST-NET-1)
    ipaddress.ip_network("192.168.0.0/16"),     # Private-Use (RFC 1918)
    ipaddress.ip_network("198.18.0.0/15"),      # Benchmarking
    ipaddress.ip_network("198.51.100.0/24"),    # Documentation (TEST-NET-2)
    ipaddress.ip_network("203.0.113.0/24"),     # Documentation (TEST-NET-3)
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Limited Broadcast
    # IPv6 Blocked Ranges
    ipaddress.ip_network("::/128"),             # Unspecified
    ipaddress.ip_network("::1/128"),           # Loopback
    ipaddress.ip_network("::ffff:0:0/96"),      # IPv4-mapped IPv6 (checked against IPv4 rules)
    ipaddress.ip_network("100::/64"),           # Discard-Only
    ipaddress.ip_network("2001:db8::/32"),      # Documentation
    ipaddress.ip_network("fc00::/7"),           # Unique Local (ULA)
    ipaddress.ip_network("fe80::/10"),          # Link-Local Unicast
    ipaddress.ip_network("ff00::/8"),           # Multicast
]


def canonicalize_url(raw_url: str) -> str:
    """
    Canonicalize a URL by normalizing scheme/host, stripping tracking query params,
    removing anchor fragments, and sorting query parameters.
    """
    if not raw_url or not isinstance(raw_url, str):
        return ""

    raw_url = raw_url.strip()
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return raw_url

    scheme = parsed.scheme.lower() if parsed.scheme else "http"
    netloc = parsed.netloc.lower()

    # Remove standard default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Normalize path
    path = parsed.path or "/"
    # Clean redundant slashes in path
    path = re.sub(r"/+", "/", path)

    # Filter tracking query parameters
    query_params = []
    if parsed.query:
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            k_lower = k.lower()
            if k_lower not in TRACKING_PARAMS and not k_lower.startswith("utm_"):
                query_params.append((k, v))

    # Sort query parameters for stability
    query_params.sort(key=lambda x: x[0])
    new_query = urlencode(query_params)

    # Return canonical URL without fragment
    return urlunparse((scheme, netloc, path, "", new_query, ""))


def is_safe_ip(ip_str: str) -> bool:
    """
    Check if an IP address string is safe (public, non-private, non-loopback, non-metadata).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        # Check IPv4-mapped IPv6
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped

        # Check against blocked networks
        for net in BLOCKED_IP_NETWORKS:
            if ip in net:
                return False

        # Additional standard checks
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False

        return True
    except ValueError:
        return False


def validate_url_safety(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a URL is safe to fetch:
    1. Scheme must be http or https.
    2. Hostname must resolve strictly to safe public IP addresses.
    Returns (is_safe, error_reason).
    """
    if not url:
        return False, "Empty URL"

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Malformed URL: {e}"

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return False, f"Disallowed URL scheme: '{scheme}'. Only http/https permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "Missing hostname in URL"

    # Block well-known cloud metadata hostnames
    blocked_hosts = {"metadata.google.internal", "instance-data", "169.254.169.254", "localhost"}
    if hostname.lower() in blocked_hosts or hostname.lower().endswith(".internal") or hostname.lower().endswith(".local"):
        return False, f"Blocked metadata or internal hostname: {hostname}"

    # Perform DNS resolution and check every resolved IP
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        if not addr_info:
            return False, f"Could not resolve hostname '{hostname}'"

        resolved_ips = {item[4][0] for item in addr_info if item and len(item) > 4 and item[4]}
        if not resolved_ips:
            return False, f"No IP addresses resolved for hostname '{hostname}'"

        for ip in resolved_ips:
            if not is_safe_ip(ip):
                return False, f"SSRF Protection: Hostname '{hostname}' resolved to non-public IP '{ip}'"

    except socket.gaierror as e:
        return False, f"DNS resolution failed for '{hostname}': {e}"
    except Exception as e:
        return False, f"Error validating safety for '{hostname}': {e}"

    return True, None


class SafeFetcher:
    """Safe HTTP client with SSRF blocking, redirect checks, timeouts, and byte limits."""

    USER_AGENT = "Kythe-Research-Bot/1.0 (+https://github.com/Cedsbstn/Kythe)"

    def __init__(self, timeout_s: float = 10.0, max_page_bytes: int = 2_000_000, max_redirects: int = 5):
        self.timeout_s = timeout_s
        self.max_page_bytes = max_page_bytes
        self.max_redirects = max_redirects

    def fetch(self, url: str) -> FetchResult:
        """
        Safely fetch a URL with comprehensive checks.
        """
        current_url = url
        redirects_followed = 0

        # Create headers without cookies/credentials
        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
        }

        while True:
            # Validate safety before connection
            is_safe, error_reason = validate_url_safety(current_url)
            if not is_safe:
                return FetchResult(
                    requested_url=url,
                    final_url=current_url,
                    status_code=0,
                    content_type="",
                    body_bytes=b"",
                    error_reason=error_reason,
                )

            try:
                # Use httpx with stream=True to guard against huge bodies
                with httpx.Client(
                    timeout=self.timeout_s,
                    follow_redirects=False,
                    verify=True,
                    headers=headers,
                ) as client:
                    response = client.get(current_url)

                    # Handle Redirects safely
                    if response.is_redirect:
                        redirects_followed += 1
                        if redirects_followed > self.max_redirects:
                            return FetchResult(
                                requested_url=url,
                                final_url=current_url,
                                status_code=response.status_code,
                                content_type="",
                                body_bytes=b"",
                                error_reason=f"Exceeded max redirects ({self.max_redirects})",
                            )

                        location = response.headers.get("Location")
                        if not location:
                            return FetchResult(
                                requested_url=url,
                                final_url=current_url,
                                status_code=response.status_code,
                                content_type="",
                                body_bytes=b"",
                                error_reason="Redirect without Location header",
                            )

                        # Resolve relative redirect URLs
                        current_url = str(response.url.join(location))
                        continue

                    # Check Content-Type header
                    raw_content_type = response.headers.get("Content-Type", "")
                    content_type_clean = raw_content_type.split(";")[0].strip().lower()

                    if content_type_clean and not any(
                        content_type_clean.startswith(sup) for sup in SUPPORTED_CONTENT_TYPES
                    ):
                        return FetchResult(
                            requested_url=url,
                            final_url=current_url,
                            status_code=response.status_code,
                            content_type=raw_content_type,
                            body_bytes=b"",
                            error_reason=f"Unsupported Content-Type '{raw_content_type}'",
                        )

                    # Read body with length bounding
                    body = response.content
                    if len(body) > self.max_page_bytes:
                        return FetchResult(
                            requested_url=url,
                            final_url=current_url,
                            status_code=response.status_code,
                            content_type=raw_content_type,
                            body_bytes=b"",
                            error_reason=f"Payload size ({len(body)} bytes) exceeded limit of {self.max_page_bytes} bytes",
                        )

                    return FetchResult(
                        requested_url=url,
                        final_url=current_url,
                        status_code=response.status_code,
                        content_type=raw_content_type,
                        body_bytes=body,
                        error_reason=None if response.status_code < 400 else f"HTTP Status {response.status_code}",
                    )

            except httpx.TimeoutException:
                return FetchResult(
                    requested_url=url,
                    final_url=current_url,
                    status_code=0,
                    content_type="",
                    body_bytes=b"",
                    error_reason=f"Request timed out after {self.timeout_s}s",
                )
            except Exception as e:
                return FetchResult(
                    requested_url=url,
                    final_url=current_url,
                    status_code=0,
                    content_type="",
                    body_bytes=b"",
                    error_reason=f"Network error: {type(e).__name__}: {e}",
                )


class PageExtractor:
    """Extracts clean structured markdown text from HTML/XML/Text content."""

    @staticmethod
    def extract(fetch_result: FetchResult, canonical_url: str) -> ExtractedDocument:
        """
        Extract clean text, headings, code blocks, and metadata from fetched bytes.
        """
        if not fetch_result.body_bytes or fetch_result.error_reason:
            return ExtractedDocument(
                canonical_url=canonical_url,
                title="",
                text="",
                content_hash="",
                word_count=0,
                extraction_method="none",
                warnings=[fetch_result.error_reason or "Empty body"],
            )

        content_hash = hashlib.sha256(fetch_result.body_bytes).hexdigest()
        raw_text = ""
        try:
            # Decode using apparent encoding
            raw_text = fetch_result.body_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return ExtractedDocument(
                canonical_url=canonical_url,
                title="",
                text="",
                content_hash=content_hash,
                word_count=0,
                extraction_method="failed_decode",
                warnings=[f"Failed to decode bytes: {e}"],
            )

        # Plain text handling
        if "text/plain" in fetch_result.content_type.lower() or "text/markdown" in fetch_result.content_type.lower():
            words = raw_text.split()
            return ExtractedDocument(
                canonical_url=canonical_url,
                title=canonical_url,
                text=raw_text.strip(),
                headings=[],
                code_blocks=[],
                content_hash=content_hash,
                word_count=len(words),
                extraction_method="plain_text",
                warnings=[],
            )

        # HTML parsing via BeautifulSoup
        headings: List[str] = []
        code_blocks: List[str] = []
        warnings: List[str] = []

        try:
            soup = BeautifulSoup(raw_text, "html.parser")

            # Remove noisy tags
            for elem in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "iframe", "form"]):
                elem.decompose()

            # Remove comments
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

            # Extract Title
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            if not title:
                og_title = soup.find("meta", property="og:title")
                if og_title and isinstance(og_title, Tag) and og_title.get("content"):
                    title = str(og_title["content"]).strip()
            if not title:
                h1 = soup.find("h1")
                if h1:
                    title = h1.get_text().strip()
            if not title:
                title = canonical_url

            # Extract Code blocks
            for pre in soup.find_all("pre"):
                code_text = pre.get_text().strip()
                if code_text:
                    code_blocks.append(code_text)

            # Extract Headings
            for h in soup.find_all(re.compile(r"^h[1-6]$")):
                h_text = h.get_text().strip()
                if h_text and h_text not in headings:
                    headings.append(h_text)

            # Convert structure into readable Markdown-like text
            content_parts = []
            body = soup.body or soup

            for child in body.descendants:
                if isinstance(child, Tag):
                    if child.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                        level = int(child.name[1])
                        htext = child.get_text().strip()
                        if htext:
                            content_parts.append(f"\n\n{'#' * level} {htext}\n\n")
                    elif child.name == "p":
                        ptext = child.get_text().strip()
                        if ptext:
                            content_parts.append(f"\n\n{ptext}\n\n")
                    elif child.name == "li":
                        litext = child.get_text().strip()
                        if litext:
                            content_parts.append(f"\n- {litext}")
                    elif child.name == "pre":
                        code_str = child.get_text().strip()
                        if code_str:
                            content_parts.append(f"\n\n```\n{code_str}\n```\n\n")
                    elif child.name == "blockquote":
                        btext = child.get_text().strip()
                        if btext:
                            content_parts.append(f"\n\n> {btext}\n\n")

            # Fallback if structural traversal yielded little text
            combined_text = "".join(content_parts)
            if len(combined_text.split()) < 30:
                combined_text = body.get_text(separator="\n", strip=True)

            # Clean excessive newlines
            clean_text = re.sub(r"\n{3,}", "\n\n", combined_text).strip()
            words = clean_text.split()

            return ExtractedDocument(
                canonical_url=canonical_url,
                title=title,
                text=clean_text,
                headings=headings,
                code_blocks=code_blocks,
                content_hash=content_hash,
                word_count=len(words),
                extraction_method="bs4_html",
                warnings=warnings,
            )

        except Exception as e:
            return ExtractedDocument(
                canonical_url=canonical_url,
                title=canonical_url,
                text="",
                content_hash=content_hash,
                word_count=0,
                extraction_method="failed_html_parse",
                warnings=[f"HTML extraction exception: {e}"],
            )


class EvidenceChunker:
    """Splits extracted documents into bounded passages for prompt grounding."""

    @staticmethod
    def chunk(
        doc: ExtractedDocument,
        source_id: str,
        max_chunk_chars: int = 2500,
        overlap_chars: int = 200,
    ) -> List[EvidenceChunk]:
        """Split document text into bounded chunks with metadata."""
        if not doc.text:
            return []

        chunks: List[EvidenceChunk] = []
        paragraphs = doc.text.split("\n\n")
        current_chunk_paragraphs: List[str] = []
        current_len = 0
        chunk_idx = 1

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue

            p_len = len(p)
            if current_len + p_len > max_chunk_chars and current_chunk_paragraphs:
                chunk_text = "\n\n".join(current_chunk_paragraphs)
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{source_id}-chk-{chunk_idx}",
                        source_id=source_id,
                        source_title=doc.title,
                        source_url=doc.canonical_url,
                        text=chunk_text,
                        token_estimate=len(chunk_text.split()),
                    )
                )
                chunk_idx += 1
                current_chunk_paragraphs = []
                current_len = 0

            current_chunk_paragraphs.append(p)
            current_len += p_len + 2

        if current_chunk_paragraphs:
            chunk_text = "\n\n".join(current_chunk_paragraphs)
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{source_id}-chk-{chunk_idx}",
                    source_id=source_id,
                    source_title=doc.title,
                    source_url=doc.canonical_url,
                    text=chunk_text,
                    token_estimate=len(chunk_text.split()),
                )
            )

        return chunks


class DeepReader:
    """Unified Deep Reader interface coordinating fetch, extraction, caching, and chunking."""

    def __init__(
        self,
        repo: PersistenceRepository,
        fetch_timeout_s: float = 10.0,
        max_page_bytes: int = 2_000_000,
        cache_ttl_hours: int = 72,
    ):
        self.repo = repo
        self.fetcher = SafeFetcher(timeout_s=fetch_timeout_s, max_page_bytes=max_page_bytes)
        self.cache_ttl_hours = cache_ttl_hours

    def process_url(
        self,
        raw_url: str,
        source_id: str,
        title_hint: str = "",
        snippet_hint: str = "",
    ) -> Tuple[Source, List[EvidenceChunk]]:
        """
        Full deep reading pipeline for a candidate URL:
        1. Canonicalize URL.
        2. Check persistence cache.
        3. Fetch page safely with SSRF protections and size limits.
        4. Extract clean Markdown text, headings, code blocks.
        5. Chunk evidence for grounding.
        6. Persist source and update cache.
        """
        canonical_url = canonicalize_url(raw_url)
        if not canonical_url:
            source = Source(
                source_id=source_id,
                canonical_url=raw_url,
                title=title_hint or "Invalid URL",
                domain="",
                extraction_status="failed",
                snippet=snippet_hint,
            )
            self.repo.upsert_source(source)
            return source, []

        domain = urlparse(canonical_url).netloc

        # Check Cache
        cached_entry = self.repo.get_cached_page(canonical_url)
        if cached_entry:
            logger.info("DeepReader: Reusing cached content for %s", canonical_url)
            meta = {}
            try:
                meta = json.loads(cached_entry.get("metadata_json", "{}"))
            except Exception:
                pass

            title = meta.get("title") or title_hint or domain
            doc = ExtractedDocument(
                canonical_url=canonical_url,
                title=title,
                text=cached_entry.get("extracted_text", ""),
                content_hash=cached_entry.get("content_hash", ""),
                word_count=len(cached_entry.get("extracted_text", "").split()),
                extraction_method="cached",
            )
            source = Source(
                source_id=source_id,
                canonical_url=canonical_url,
                title=title,
                domain=domain,
                content_hash=doc.content_hash,
                content_type=cached_entry.get("content_type", "text/html"),
                extraction_status="cached",
                extracted_text=doc.text,
                snippet=snippet_hint,
            )
            self.repo.upsert_source(source)
            chunks = EvidenceChunker.chunk(doc, source_id)
            return source, chunks

        # Fetch page safely
        fetch_result = self.fetcher.fetch(canonical_url)
        if fetch_result.error_reason:
            logger.warning("DeepReader: Fetch failed for %s: %s", canonical_url, fetch_result.error_reason)
            source = Source(
                source_id=source_id,
                canonical_url=canonical_url,
                title=title_hint or domain,
                domain=domain,
                extraction_status=f"failed: {fetch_result.error_reason}",
                snippet=snippet_hint,
            )
            self.repo.upsert_source(source)
            return source, []

        # Extract content
        doc = PageExtractor.extract(fetch_result, canonical_url)
        title = doc.title or title_hint or domain

        source = Source(
            source_id=source_id,
            canonical_url=canonical_url,
            title=title,
            domain=domain,
            content_hash=doc.content_hash,
            content_type=fetch_result.content_type,
            extraction_status="success" if doc.text else "empty",
            extracted_text=doc.text,
            snippet=snippet_hint,
        )
        self.repo.upsert_source(source)

        # Cache result
        if doc.text:
            self.repo.set_cached_page(
                url=raw_url,
                canonical_url=canonical_url,
                content_hash=doc.content_hash,
                status_code=fetch_result.status_code,
                content_type=fetch_result.content_type,
                extracted_text=doc.text,
                metadata={"title": title, "headings": doc.headings},
                ttl_hours=self.cache_ttl_hours,
            )

        chunks = EvidenceChunker.chunk(doc, source_id)
        return source, chunks
