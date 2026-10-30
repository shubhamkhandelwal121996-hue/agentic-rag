import ipaddress
import logging
import os
import re
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ----------------------------------HTTP requests and URL validation----------------------------------------- #
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

REQUEST_TIMEOUT = 30            # web and arXiv API requests
PDF_TIMEOUT = 60                # PDF downloads
MAX_RETRIES = 2                 # retries for temporary request failures
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
USER_AGENT = "AgenticRAG/1.0"

ALLOW_PRIVATE_URLS = os.getenv("ALLOW_PRIVATE_URLS", "").lower() in {"1", "true", "yes"}

# Minimum text length used to flag potentially scanned pages.
MIN_PAGE_CHARS = 50

ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM = {"atom": "http://www.w3.org/2005/Atom"}

# Keep only the metadata needed for retrieval and source tracking.
KEPT_METADATA = ("page", "start_index")

# Match both modern and legacy arXiv IDs without version suffixes.
_ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})"
ARXIV_ID_PATTERN = re.compile(rf"(?<![\w.])({_ARXIV_ID})(?:v\d+)?\b")
ARXIV_URL_PATTERN = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", re.I)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    language="markdown",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)

# ---------------------------------HTTP requests and URL validation------------------------------------------ #
def _assert_public_url(url: str) -> None:
    """Reject URLs that resolve to non-public IP addresses."""
    if ALLOW_PRIVATE_URLS:
        return

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Unsupported URL: {url}")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as error:
        raise ValueError(f"Could not resolve host: {parsed.hostname}") from error

    for info in addresses:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError(f"Refusing to fetch non-public address: {parsed.hostname}")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Check redirected URLs before following them."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _assert_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirectHandler)


def _http_get(url: str, timeout: int, max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    """Fetch a URL with retries, timeouts, and download limits."""
    _assert_public_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(MAX_RETRIES + 1):
        try:
            with _opener.open(request, timeout=timeout) as response:
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"Download exceeds {max_bytes // (1024 * 1024)} MB limit: {url}")
            return data
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == MAX_RETRIES:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == MAX_RETRIES:
                raise
        time.sleep(2 ** attempt)

    raise RuntimeError("unreachable") 


# ----------------------------------Text preprocessing and chunking----------------------------------------- #
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")   # join words split across lines
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """Clean text extracted from PDFs and web pages."""
    text = text.replace("\x00", "")
    text = _HYPHEN_BREAK.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _split_and_tag(
    documents: list[Document],
    title: str,
    source: str,
    source_type: str,
    splitter: RecursiveCharacterTextSplitter = _splitter,
) -> list[Document]:
    """Split documents into chunks and attach source metadata."""
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]

    if not chunks:
        # Avoid indexing documents with no extractable text.
        raise ValueError(
            f"No extractable text found in: {source} "
            "(scanned PDF, empty file, or a page rendered by JavaScript?)"
        )

    for chunk in chunks:
        kept = {k: chunk.metadata[k] for k in KEPT_METADATA if k in chunk.metadata}
        chunk.metadata = {**kept, "title": title, "source": source, "source_type": source_type}

    logger.info("Loaded %d chunks from %s", len(chunks), source)
    return chunks


# -----------------------------------Text and Markdown files---------------------------------------- #
def _text_loader(path: str) -> TextLoader:
    return TextLoader(path, encoding="utf-8", autodetect_encoding=True)


def _load_text_file(
    file_path: str,
    source_type: str,
    splitter: RecursiveCharacterTextSplitter = _splitter,
) -> list[Document]:
    path = Path(file_path)
    documents = _text_loader(str(path)).load()
    return _split_and_tag(documents, path.stem, str(path), source_type, splitter)


def load_text(file_path: str) -> list[Document]:
    return _load_text_file(file_path, "text")


def load_markdown(file_path: str) -> list[Document]:
    return _load_text_file(file_path, "markdown", _md_splitter)


# ------------------------------------PDF loading--------------------------------------- #
_JUNK_TITLE = re.compile(r"(\.(dvi|docx?|tex|pdf)$)|^(microsoft word|untitled)", re.I)


def _usable_pdf_title(documents: list[Document]) -> str | None:
    """Return a PDF title when its metadata is usable."""
    title = (documents[0].metadata.get("title") or "").strip()
    return title if len(title) >= 8 and not _JUNK_TITLE.search(title) else None


def _warn_if_mostly_scanned(documents: list[Document], source: str) -> None:
    empty = sum(1 for d in documents if len(d.page_content.strip()) < MIN_PAGE_CHARS)
    if empty / len(documents) > 0.5:
        logger.warning(
            "%d of %d pages in %s have almost no text; it is probably scanned and needs OCR",
            empty, len(documents), source,
        )


def _finalize_pdf(
    documents: list[Document],
    source: str,
    source_type: str,
    title: str | None,
    fallback_title: str,
) -> list[Document]:
    if not documents:
        raise ValueError(f"No readable pages found in: {source}")

    _warn_if_mostly_scanned(documents, source)

    for document in documents:
        document.page_content = _clean_text(document.page_content)

    # Prefer the supplied title, then PDF metadata, then the fallback.
    title = title or _usable_pdf_title(documents) or fallback_title
    return _split_and_tag(documents, title, source, source_type)


def load_pdf(file_path: str) -> list[Document]:
    path = Path(file_path)
    documents = PyMuPDFLoader(str(path)).load()
    # Use the filename as the title for local PDFs.
    return _finalize_pdf(documents, str(path), "pdf", title=path.stem, fallback_title=path.stem)


def _load_pdf_bytes(
    pdf_bytes: bytes,
    source: str,
    source_type: str,
    title: str | None,
    fallback_title: str,
) -> list[Document]:
    if not pdf_bytes.startswith(b"%PDF"):
        # Reject HTML responses returned instead of a PDF.
        raise ValueError(f"Response from {source} was not a PDF (rate-limited or withdrawn?)")

    # The temporary PDF is removed after loading.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        pdf_path = Path(tmp_dir) / "paper.pdf"
        pdf_path.write_bytes(pdf_bytes)
        documents = PyMuPDFLoader(str(pdf_path)).load()

    return _finalize_pdf(documents, source, source_type, title, fallback_title)


# ----------------------------------Web pages----------------------------------------- # 

_BOILERPLATE_TAGS = ["script", "style", "noscript", "nav", "footer", "aside", "form", "svg", "iframe"]
_BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "blockquote",
               "td", "th", "figcaption", "dd", "dt"]


def _html_to_text(raw_html: bytes, url: str) -> tuple[str, str]:
    """Extract the page title and readable text from HTML."""
    soup = BeautifulSoup(raw_html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup

    # Skip nested blocks to avoid duplicate text.
    blocks = []
    for element in root.find_all(_BLOCK_TAGS):
        if element.find_parent(_BLOCK_TAGS):
            continue
        text = element.get_text(" ", strip=True)
        if text:
            blocks.append(text)

    text = "\n\n".join(blocks) or root.get_text(" ", strip=True)
    return title or url, text


def load_webpage(url: str) -> list[Document]:
    raw = _http_get(url, REQUEST_TIMEOUT)

    # Handle direct PDF links separately from HTML pages.
    if raw.startswith(b"%PDF"):
        return _load_pdf_bytes(raw, url, "web_pdf", title=None, fallback_title=url)

    title, text = _html_to_text(raw, url)
    document = Document(page_content=_clean_text(text), metadata={})
    return _split_and_tag([document], title, url, "web")


# ---------------------------------arXiv------------------------------------------ #
def _extract_arxiv_id(text: str) -> str | None:
    """Extract an arXiv ID from a link or text."""
    match = ARXIV_ID_PATTERN.search(text)
    return match.group(1) if match else None


def _arxiv_query(params: dict[str, str | int]) -> tuple[str, str] | None:
    """Get the first matching paper ID and title from arXiv."""
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    root = ET.fromstring(_http_get(url, REQUEST_TIMEOUT))

    entry = root.find("atom:entry", _ATOM)
    if entry is None:
        return None

    raw_id = entry.findtext("atom:id", "", _ATOM)
    if "/api/errors" in raw_id:  # arXiv reports a bad ID as a fake entry
        return None

    arxiv_id = _extract_arxiv_id(raw_id)
    if arxiv_id is None:
        return None

    title = " ".join(entry.findtext("atom:title", "", _ATOM).split())
    return arxiv_id, title or arxiv_id


def _resolve_arxiv(query: str) -> tuple[str, str | None]:
    """Resolve an arXiv ID, URL, or title to paper metadata."""
    arxiv_id = _extract_arxiv_id(query)

    if arxiv_id:
        try:
            found = _arxiv_query({"id_list": arxiv_id, "max_results": 1})
        except (urllib.error.URLError, TimeoutError, ET.ParseError):
            # Continue if metadata is unavailable but the PDF can be fetched.
            return arxiv_id, None
        if found is None:
            raise ValueError(f"No ArXiv paper found for ID: {arxiv_id}")
        return found

    phrase = query.strip().strip('"')
    for field in ("ti", "all"):  # search title before other fields
        found = _arxiv_query(
            {"search_query": f'{field}:"{phrase}"', "max_results": 1, "sortBy": "relevance"}
        )
        if found:
            return found

    raise ValueError(f"No ArXiv paper found for: {query}")


def _download_arxiv_pdf(arxiv_id: str) -> bytes:
    return _http_get(f"https://arxiv.org/pdf/{arxiv_id}", PDF_TIMEOUT)


def load_arxiv(query: str) -> list[Document]:
    """Download and process a paper from arXiv."""
    known_id = _extract_arxiv_id(query)

    if known_id:
        # Fetch metadata and the PDF concurrently when the ID is known.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pdf_future = pool.submit(_download_arxiv_pdf, known_id)
            arxiv_id, title = _resolve_arxiv(query)  # raises if the ID doesn't exist
            pdf_bytes = pdf_future.result()
    else:
        arxiv_id, title = _resolve_arxiv(query)
        pdf_bytes = _download_arxiv_pdf(arxiv_id)

    return _load_pdf_bytes(
        pdf_bytes,
        source=f"https://arxiv.org/abs/{arxiv_id}",  # permanent source URL
        source_type="arxiv",
        title=title,                                  # prefer arXiv metadata
        fallback_title=arxiv_id,
    )


# ----------------------------------Document loader selection----------------------------------------- #
_FILE_LOADERS: dict[str, Callable[[str], list[Document]]] = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
}


def load_document(source: str) -> list[Document]:
    """Load and split a supported document or URL."""
    source = source.strip()

    if source.startswith(("https://", "http://")):
        # Load arXiv PDFs rather than their abstract pages.
        return load_arxiv(source) if ARXIV_URL_PATTERN.match(source) else load_webpage(source)

    path = Path(source)
    loader = _FILE_LOADERS.get(path.suffix.lower())

    if loader is None:
        raise ValueError(f"Unsupported file extension: {path.suffix.lower() or '(none)'}")
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    return loader(source)