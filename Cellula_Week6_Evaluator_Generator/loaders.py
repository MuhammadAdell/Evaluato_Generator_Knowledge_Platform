import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go",
    ".rs", ".rb", ".php", ".sql", ".sh", ".html", ".css",
    ".json", ".yaml", ".yml", ".ipynb",
}

CODE_LANGUAGES = {".py": "python", ".js": "javascript", ".ts": "typescript", ".cs": "csharp"}

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def load_pdf(path: Path) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    documents = PyPDFLoader(str(path)).load()
    for document in documents:
        document.metadata = {
            "source": path.name,
            "type": "pdf",
            "page": int(document.metadata.get("page", 0)) + 1,
        }
    return documents


def load_docx(path: Path) -> List[Document]:
    from docx import Document as DocxDocument

    docx = DocxDocument(str(path))
    lines = [paragraph.text.strip() for paragraph in docx.paragraphs if paragraph.text.strip()]
    for table in docx.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [Document(page_content="\n".join(lines), metadata={"source": path.name, "type": "docx"})]


def load_pptx(path: Path) -> List[Document]:
    from pptx import Presentation

    documents = []
    for number, slide in enumerate(Presentation(str(path)).slides, start=1):
        parts = [
            shape.text_frame.text.strip()
            for shape in slide.shapes
            if shape.has_text_frame and shape.text_frame.text.strip()
        ]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append(slide.notes_slide.notes_text_frame.text.strip())
        if parts:
            documents.append(
                Document(
                    page_content="\n".join(parts),
                    metadata={"source": path.name, "type": "pptx", "page": number},
                )
            )
    return documents


def load_text(path: Path) -> List[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8", errors="replace"),
            metadata={"source": path.name, "type": "txt"},
        )
    ]


def load_code(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    return [
        Document(
            page_content=path.read_text(encoding="utf-8", errors="replace"),
            metadata={
                "source": path.name,
                "type": "code",
                "language": CODE_LANGUAGES.get(suffix, suffix.lstrip(".")),
            },
        )
    ]


def load_url(url: str) -> List[Document]:
    response = requests.get(
        url,
        timeout=25,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n")).strip()
    title = soup.title.get_text(strip=True) if soup.title else url
    return [
        Document(
            page_content=text,
            metadata={"source": title, "type": "web", "url": url},
        )
    ]


def load_wikipedia(value: str) -> List[Document]:
    title = value.strip()
    if "wikipedia.org/wiki/" in value:
        title = unquote(value.split("/wiki/")[-1]).replace("_", " ")
    response = requests.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "format": "json",
            "titles": title,
        },
        timeout=25,
        headers={"User-Agent": "cellula-project3/1.0 (student project)"},
    )
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    if "missing" in page:
        raise ValueError(f"Wikipedia page not found: {value}")
    text = page.get("extract", "").strip()
    if not text:
        raise ValueError(f"Wikipedia page has no readable text: {value}")
    return [
        Document(
            page_content=text,
            metadata={
                "source": page["title"],
                "type": "wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
            },
        )
    ]


def load_wav(path: Path) -> List[Document]:
    worker = Path(__file__).with_name("stt_worker.py")
    result = subprocess.run(
        [sys.executable, str(worker), str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip()[-300:] or "Speech-to-text worker failed.")
    transcript = json.loads(result.stdout)
    return [
        Document(
            page_content=transcript["text"],
            metadata={
                "source": path.name,
                "type": "audio",
                "language": transcript["language"],
                "duration_seconds": transcript["duration_seconds"],
            },
        )
    ]


def load_file(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".pptx":
        return load_pptx(path)
    if suffix == ".wav":
        return load_wav(path)
    if suffix in CODE_EXTENSIONS:
        return load_code(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    if suffix == ".ppt":
        raise ValueError("Old .ppt files are not supported. Save the file as .pptx first.")
    raise ValueError(f"Unsupported file type: {suffix or 'no extension'}")


def empty_check(documents: List[Document], name: str) -> None:
    if not any(document.page_content.strip() for document in documents):
        raise ValueError(f"No text could be extracted from {name}.")
