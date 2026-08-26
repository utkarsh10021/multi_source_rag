import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, MAX_PAGES, MAX_LINKS_PER_PAGE,
    SUPPORTED_EXTENSIONS
)

def clean_url(url: str) -> str:
    url = url.strip().rstrip("\\")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()

def is_same_domain(base_url: str, new_url: str) -> bool:
    a = urlparse(base_url).netloc.lower().replace("www.", "")
    b = urlparse(new_url).netloc.lower().replace("www.", "")
    return a == b

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

def crawl_website(start_url: str) -> list[Document]:
    start_url = clean_url(start_url)
    visited, queue, documents = set(), [start_url], []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        while queue and len(documents) < MAX_PAGES:
            current = clean_url(queue.pop(0))
            if current in visited or not is_same_domain(start_url, current):
                continue
            visited.add(current)

            try:
                page.goto(current, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(1500)
                title = page.title()

                try:
                    page.evaluate("""
                        () => {
                            document.querySelectorAll(
                                'script,style,noscript,iframe,svg'
                            ).forEach(e => e.remove());
                        }
                    """)
                except Exception:
                    pass

                text = clean_text(page.locator("body").inner_text())
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": current,
                            "title": title or current,
                            "type": "website",
                            "url": current,
                        },
                    ))

                discovered = 0
                for link in page.locator("a").all():
                    if discovered >= MAX_LINKS_PER_PAGE:
                        break
                    try:
                        href = link.get_attribute("href")
                    except Exception:
                        continue
                    if not href or href.startswith(
                        ("javascript:", "mailto:", "tel:", "#")
                    ):
                        continue
                    absolute = clean_url(urljoin(current, href))
                    if (
                        absolute.startswith(("http://", "https://"))
                        and is_same_domain(start_url, absolute)
                        and absolute not in visited
                        and absolute not in queue
                    ):
                        queue.append(absolute)
                        discovered += 1

            except PlaywrightTimeoutError:
                continue
            except Exception:
                continue

        browser.close()

    return documents

def load_pdf(file_path: str) -> list[Document]:
    docs = PyPDFLoader(file_path).load()
    return _set_file_metadata(docs, file_path, "pdf")

def load_docx(file_path: str) -> list[Document]:
    docs = Docx2txtLoader(file_path).load()
    return _set_file_metadata(docs, file_path, "docx")

def load_csv(file_path: str) -> list[Document]:
    docs = CSVLoader(file_path=file_path).load()
    return _set_file_metadata(docs, file_path, "csv")

def _set_file_metadata(docs, file_path, source_type):
    for doc in docs:
        doc.metadata.update({
            "source": str(Path(file_path).resolve()),
            "title": Path(file_path).name,
            "type": source_type,
        })
    return docs

def load_file(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext == ".pdf":
        return load_pdf(str(path))
    if ext == ".docx":
        return load_docx(str(path))
    return load_csv(str(path))

def create_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
