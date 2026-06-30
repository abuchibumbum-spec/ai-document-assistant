#!/usr/bin/env python3
"""Local AI document assistant.

Upload notes or documents, index them locally, search with a hybrid semantic
score, ask cited questions, and summarize matching sections.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


APP_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = APP_DIR / "documents"
DEFAULT_DB = APP_DIR / "documents.db"
CHUNK_WORDS = 220
CHUNK_OVERLAP = 55
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".html", ".htm", ".pdf"}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "has", "have", "he", "her", "his", "how", "i", "in", "is", "it", "its",
    "may", "of", "on", "or", "our", "she", "should", "that", "the", "their",
    "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
}


@dataclass(frozen=True)
class SearchResult:
    document: str
    title: str
    chunk_id: int
    text: str
    score: float


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            kind TEXT NOT NULL,
            modified REAL NOT NULL,
            extracted_ok INTEGER NOT NULL DEFAULT 1,
            message TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_number INTEGER NOT NULL,
            text TEXT NOT NULL,
            tokens TEXT NOT NULL,
            shingles TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
    return conn


def migrate_schema(db_path: Path = DEFAULT_DB) -> None:
    conn = connect(db_path)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
    chunk_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    with conn:
        if "kind" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN kind TEXT NOT NULL DEFAULT 'file'")
        if "extracted_ok" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN extracted_ok INTEGER NOT NULL DEFAULT 1")
        if "message" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN message TEXT NOT NULL DEFAULT ''")
        if "shingles" not in chunk_columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN shingles TEXT NOT NULL DEFAULT ''")
    conn.close()


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._ -]+", "_", name).strip(" .")
    return cleaned or "document.txt"


def unique_path(folder: Path, filename: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    filename = safe_name(filename)
    candidate = folder / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for number in range(2, 1000):
        candidate = folder / f"{stem}-{number}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not create a unique filename")


def extract_pdf(path: Path) -> tuple[str, bool, str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            return (
                "",
                False,
                "PDF upload saved, but PDF text extraction needs pypdf or PyPDF2 installed.",
            )

    try:
        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"Page {index}. {page_text}")
        text = "\n\n".join(pages).strip()
        if not text:
            return "", False, "No selectable text found in this PDF."
        return text, True, ""
    except Exception as exc:
        return "", False, f"Could not extract text from PDF: {exc}"


def read_document(path: Path) -> tuple[str, bool, str]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return "", False, f"Unsupported file type: {suffix}"
    if suffix == ".pdf":
        return extract_pdf(path)

    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.I)
        text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    if suffix == ".md":
        text = re.sub(r"^#{1,6}\s*(.+)$", r"\1.", text, flags=re.MULTILINE)
        text = re.sub(r"[*_`>]", " ", text)
    return re.sub(r"\s+", " ", text).strip(), True, ""


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9']+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    ]


def shingles(text: str) -> list[str]:
    compact = re.sub(r"[^a-z0-9]+", " ", text.lower())
    words = [word for word in compact.split() if word not in STOP_WORDS]
    values: list[str] = []
    for size in (2, 3):
        values.extend(" ".join(words[i : i + size]) for i in range(max(0, len(words) - size + 1)))
    return values


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + CHUNK_WORDS]).strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def find_documents(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def index_documents(folder: Path = DOCUMENTS_DIR, db_path: Path = DEFAULT_DB) -> dict[str, Any]:
    migrate_schema(db_path)
    folder.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    indexed_docs = 0
    indexed_chunks = 0
    warnings: list[str] = []

    with conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        for path in find_documents(folder):
            text, ok, message = read_document(path)
            relative_path = str(path.relative_to(folder))
            cursor = conn.execute(
                """
                INSERT INTO documents (path, title, kind, modified, extracted_ok, message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (relative_path, path.stem, path.suffix.lower().lstrip("."), path.stat().st_mtime, int(ok), message),
            )
            document_id = cursor.lastrowid
            if message:
                warnings.append(f"{relative_path}: {message}")
            for number, chunk in enumerate(chunk_text(text), start=1):
                conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_number, text, tokens, shingles)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, number, chunk, " ".join(tokenize(chunk)), " || ".join(shingles(chunk))),
                )
                indexed_chunks += 1
            indexed_docs += 1
    conn.close()
    return {"documents": indexed_docs, "chunks": indexed_chunks, "warnings": warnings}


def load_chunks(db_path: Path = DEFAULT_DB) -> list[sqlite3.Row]:
    migrate_schema(db_path)
    conn = connect(db_path)
    rows = conn.execute(
        """
        SELECT chunks.chunk_number, chunks.text, chunks.tokens, chunks.shingles,
               documents.path, documents.title
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        ORDER BY documents.path, chunks.chunk_number
        """
    ).fetchall()
    conn.close()
    return rows


def document_status(db_path: Path = DEFAULT_DB) -> list[dict[str, Any]]:
    migrate_schema(db_path)
    conn = connect(db_path)
    rows = conn.execute(
        """
        SELECT documents.path, documents.title, documents.kind, documents.extracted_ok,
               documents.message, COUNT(chunks.id) AS chunks
        FROM documents
        LEFT JOIN chunks ON chunks.document_id = documents.id
        GROUP BY documents.id
        ORDER BY documents.path
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def hybrid_search(query: str, db_path: Path = DEFAULT_DB, limit: int = 6) -> list[SearchResult]:
    query_tokens = tokenize(query)
    query_shingles = set(shingles(query))
    if not query_tokens and not query_shingles:
        return []

    rows = load_chunks(db_path)
    total_chunks = max(1, len(rows))
    token_document_frequency: dict[str, int] = defaultdict(int)
    shingle_document_frequency: dict[str, int] = defaultdict(int)
    token_counts: list[Counter[str]] = []
    shingle_sets: list[set[str]] = []

    for row in rows:
        counts = Counter(row["tokens"].split())
        token_counts.append(counts)
        for token in set(counts):
            token_document_frequency[token] += 1
        chunk_shingles = set(filter(None, row["shingles"].split(" || ")))
        shingle_sets.append(chunk_shingles)
        for item in chunk_shingles:
            shingle_document_frequency[item] += 1

    results: list[SearchResult] = []
    for row, counts, chunk_shingles in zip(rows, token_counts, shingle_sets):
        token_score = 0.0
        chunk_length = sum(counts.values()) or 1
        for token in query_tokens:
            frequency = counts[token]
            if frequency:
                inverse_document_frequency = math.log((1 + total_chunks) / (1 + token_document_frequency[token])) + 1
                token_score += (frequency / chunk_length) * inverse_document_frequency

        semantic_score = 0.0
        if query_shingles and chunk_shingles:
            overlap = query_shingles & chunk_shingles
            semantic_score = len(overlap) / max(1, min(len(query_shingles), len(chunk_shingles)))

        lower_text = row["text"].lower()
        phrase_score = 0.07 if query.lower() in lower_text else 0.0
        soft_overlap = sum(1 for token in query_tokens if token in lower_text) / max(1, len(query_tokens))
        score = token_score + (semantic_score * 0.75) + (soft_overlap * 0.15) + phrase_score

        if score > 0:
            results.append(
                SearchResult(
                    document=row["path"],
                    title=row["title"],
                    chunk_id=row["chunk_number"],
                    text=row["text"],
                    score=score,
                )
            )
    return sorted(results, key=lambda result: result.score, reverse=True)[:limit]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def best_sentences(query: str, results: list[SearchResult], max_sentences: int = 3) -> list[tuple[str, SearchResult]]:
    query_tokens = set(tokenize(query))
    ranked: list[tuple[float, str, SearchResult]] = []
    for result in results:
        for sentence in split_sentences(result.text):
            sentence_tokens = set(tokenize(sentence))
            if len(sentence_tokens) < 4:
                continue
            overlap = len(query_tokens & sentence_tokens)
            if overlap:
                coverage = overlap / max(1, len(query_tokens))
                concise_bonus = 1 / max(14, len(sentence.split()))
                ranked.append((coverage + result.score + concise_bonus, sentence, result))
    ranked.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[str, SearchResult]] = []
    seen = set()
    best_score = ranked[0][0] if ranked else 0
    for score, sentence, result in ranked:
        if selected and score < best_score * 0.9:
            break
        normalized = sentence.lower()
        if normalized in seen:
            continue
        selected.append((sentence, result))
        seen.add(normalized)
        if len(selected) >= max_sentences:
            break
    return selected


def answer_question(question: str, db_path: Path = DEFAULT_DB, limit: int = 6) -> dict[str, Any]:
    results = hybrid_search(question, db_path, limit)
    if not results:
        return {
            "answer": "I could not find a relevant answer in the indexed documents.",
            "citations": [],
            "results": [],
        }

    selected = best_sentences(question, results, 3)
    if selected:
        answer = " ".join(sentence for sentence, _ in selected)
        cited_results = []
        seen = set()
        for _, result in selected:
            key = (result.document, result.chunk_id)
            if key not in seen:
                cited_results.append(result)
                seen.add(key)
    else:
        answer = results[0].text[:850]
        cited_results = [results[0]]

    return {
        "answer": answer,
        "citations": [
            {"document": result.document, "chunk": result.chunk_id, "score": round(result.score, 4)}
            for result in cited_results
        ],
        "results": [
            {
                "document": result.document,
                "title": result.title,
                "chunk": result.chunk_id,
                "score": round(result.score, 4),
                "text": result.text,
            }
            for result in results
        ],
    }


def summarize_text(text: str, focus: str = "", max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "No text available to summarize."
    focus_tokens = set(tokenize(focus))
    all_tokens = tokenize(text)
    frequencies = Counter(all_tokens)

    ranked: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        tokens = tokenize(sentence)
        if not tokens:
            continue
        score = sum(frequencies[token] for token in tokens) / len(tokens)
        if focus_tokens:
            score += len(focus_tokens & set(tokens)) * 2.5
        score += 0.25 / (index + 1)
        ranked.append((score, index, sentence))

    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen = sorted(ranked[:max_sentences], key=lambda item: item[1])
    return " ".join(sentence for _, _, sentence in chosen)


def summarize(query: str = "", db_path: Path = DEFAULT_DB, limit: int = 4) -> dict[str, Any]:
    if query:
        results = hybrid_search(query, db_path, limit)
    else:
        rows = load_chunks(db_path)[:limit]
        results = [
            SearchResult(row["path"], row["title"], row["chunk_number"], row["text"], 0)
            for row in rows
        ]
    if not results:
        return {"summary": "No indexed text is available to summarize.", "citations": []}

    combined = " ".join(result.text for result in results)
    summary = summarize_text(combined, query)
    return {
        "summary": summary,
        "citations": [
            {"document": result.document, "chunk": result.chunk_id, "score": round(result.score, 4)}
            for result in results
        ],
    }


def create_note(title: str, body: str, folder: Path = DOCUMENTS_DIR) -> Path:
    title = title.strip() or "Untitled Note"
    body = body.strip()
    path = unique_path(folder, f"{title}.md")
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def create_sample_documents(folder: Path = DOCUMENTS_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "remote_work_policy.md").write_text(
        """# Remote Work Policy

Employees may work remotely up to three days per week with manager approval.
Core collaboration hours are 10:00 AM to 3:00 PM Eastern Time. Team members
should keep calendars current and join required meetings with working audio.

Remote employees can request a one-time home office reimbursement of up to
$600 for a chair, desk, monitor, keyboard, or other approved equipment.
Receipts must be submitted within 30 days of purchase.
""",
        encoding="utf-8",
    )
    (folder / "expense_guidelines.md").write_text(
        """# Expense Guidelines

Travel meals are reimbursable up to $75 per day when an employee is away from
their home office overnight. Alcohol is not reimbursable. Airfare should be
booked in economy class unless an exception is approved by finance.

Expense reports are due by the fifth business day of the following month.
Reports need a business purpose, receipt image, amount, date, and category.
""",
        encoding="utf-8",
    )
    (folder / "benefits_summary.txt").write_text(
        """Benefits Summary

Full-time employees receive medical, dental, and vision coverage. The company
matches 401(k) contributions up to 4 percent of eligible compensation after
90 days of employment.

Paid time off starts at 15 days per year and increases after three years of
service. Sick time is tracked separately from vacation time.
""",
        encoding="utf-8",
    )


def parse_multipart(body: bytes, content_type: str) -> dict[str, list[dict[str, Any]]]:
    match = re.search(r"boundary=(.+)", content_type)
    if not match:
        return {}
    boundary = match.group(1).strip().strip('"').encode()
    parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_part in body.split(b"--" + boundary):
        raw_part = raw_part.strip(b"\r\n")
        if not raw_part or raw_part == b"--":
            continue
        header_blob, _, value = raw_part.partition(b"\r\n\r\n")
        headers = header_blob.decode("utf-8", errors="ignore")
        disposition = re.search(r'name="([^"]+)"(?:;\s*filename="([^"]*)")?', headers)
        if not disposition:
            continue
        name = disposition.group(1)
        filename = disposition.group(2) or ""
        parts[name].append({"filename": filename, "content": value.rstrip(b"\r\n")})
    return parts


def render_home() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Document Assistant</title>
  <style>
    :root {
      --ink: #202426;
      --muted: #667274;
      --line: #d8dfdd;
      --paper: #f5f6f2;
      --panel: #ffffff;
      --accent: #167064;
      --accent-2: #2f5f91;
      --warn: #9a5b14;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font-family: Arial, Helvetica, sans-serif; }
    main { width: min(1180px, calc(100% - 32px)); margin: 28px auto; }
    header { display: flex; justify-content: space-between; gap: 18px; align-items: end; border-bottom: 1px solid var(--line); padding-bottom: 18px; }
    h1 { margin: 0; font-size: 32px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
    .muted { color: var(--muted); }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 18px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; }
    input, textarea {
      width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px;
      font: inherit; background: white; min-height: 46px;
    }
    textarea { min-height: 132px; resize: vertical; line-height: 1.45; }
    button {
      min-height: 42px; border: 0; border-radius: 8px; padding: 0 16px; background: var(--accent);
      color: white; font-weight: 700; cursor: pointer;
    }
    button.secondary { background: var(--accent-2); }
    .row { display: flex; gap: 10px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
    .askbar { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; margin: 22px 0 16px; }
    .answer { font-size: 18px; line-height: 1.55; }
    .citation { display: inline-block; margin: 8px 8px 0 0; padding: 6px 9px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 13px; }
    .result, .doc { border-top: 1px solid var(--line); padding-top: 14px; margin-top: 14px; }
    .result:first-child, .doc:first-child { border-top: 0; margin-top: 0; padding-top: 0; }
    pre { white-space: pre-wrap; font-family: Arial, Helvetica, sans-serif; line-height: 1.45; color: #344043; margin-bottom: 0; }
    .warn { color: var(--warn); }
    @media (max-width: 820px) {
      header, .grid, .askbar { display: block; }
      section, button { margin-top: 10px; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>AI Document Assistant</h1>
        <div class="muted">Upload documents, add notes, search, ask, and summarize.</div>
      </div>
      <button class="secondary" onclick="reindex()">Reindex</button>
    </header>

    <div class="grid">
      <section>
        <h2>Upload PDF or File</h2>
        <input id="file" type="file" accept=".pdf,.txt,.md,.csv,.json,.html,.htm">
        <div class="row"><button onclick="uploadFile()">Upload</button><span id="uploadStatus" class="muted"></span></div>
      </section>

      <section>
        <h2>Add Note</h2>
        <input id="noteTitle" placeholder="Note title">
        <textarea id="noteBody" placeholder="Paste notes here..."></textarea>
        <div class="row"><button onclick="saveNote()">Save Note</button><span id="noteStatus" class="muted"></span></div>
      </section>
    </div>

    <div class="askbar">
      <input id="question" placeholder="Ask or search across your documents..." autofocus>
      <button onclick="ask()">Ask</button>
      <button class="secondary" onclick="summarize()">Summarize</button>
    </div>

    <section>
      <h2>Answer or Summary</h2>
      <div id="answer" class="answer muted">Your answer will appear here.</div>
      <div id="citations"></div>
    </section>

    <div class="grid">
      <section>
        <h2>Source Matches</h2>
        <div id="results" class="muted">No search yet.</div>
      </section>
      <section>
        <h2>Indexed Library</h2>
        <div id="docs" class="muted">Loading...</div>
      </section>
    </div>
  </main>

  <script>
    const question = document.getElementById("question");
    question.addEventListener("keydown", event => { if (event.key === "Enter") ask(); });
    refreshDocs();

    async function ask() {
      const q = question.value.trim();
      if (!q) return;
      setAnswer("Searching...", true);
      const data = await getJson(`/api/ask?q=${encodeURIComponent(q)}`);
      renderAnswer(data.answer, data.citations);
      renderResults(data.results);
    }

    async function summarize() {
      const q = question.value.trim();
      setAnswer("Summarizing...", true);
      const data = await getJson(`/api/summarize?q=${encodeURIComponent(q)}`);
      renderAnswer(data.summary, data.citations);
    }

    async function uploadFile() {
      const file = document.getElementById("file").files[0];
      if (!file) return;
      const form = new FormData();
      form.append("file", file);
      document.getElementById("uploadStatus").textContent = "Uploading...";
      const response = await fetch("/api/upload", { method: "POST", body: form });
      const data = await response.json();
      document.getElementById("uploadStatus").textContent = data.message;
      await refreshDocs();
    }

    async function saveNote() {
      const title = document.getElementById("noteTitle").value;
      const body = document.getElementById("noteBody").value;
      if (!body.trim()) return;
      const response = await fetch("/api/note", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, body })
      });
      const data = await response.json();
      document.getElementById("noteStatus").textContent = data.message;
      document.getElementById("noteBody").value = "";
      await refreshDocs();
    }

    async function reindex() {
      setAnswer("Reindexing...", true);
      const data = await getJson("/api/reindex", { method: "POST" });
      setAnswer(`Indexed ${data.documents} document(s) and ${data.chunks} section(s).`, true);
      renderWarnings(data.warnings || []);
      await refreshDocs();
    }

    async function refreshDocs() {
      const data = await getJson("/api/documents");
      document.getElementById("docs").innerHTML = data.documents.map(item => `
        <div class="doc">
          <strong>${escapeHtml(item.path)}</strong>
          <div class="muted">${item.kind.toUpperCase()} - ${item.chunks} indexed section(s)</div>
          ${item.message ? `<div class="warn">${escapeHtml(item.message)}</div>` : ""}
        </div>
      `).join("") || "No documents indexed yet.";
    }

    function renderAnswer(text, citations) {
      setAnswer(text, false);
      document.getElementById("citations").innerHTML = (citations || []).map(item =>
        `<span class="citation">${escapeHtml(item.document)} / section ${item.chunk}</span>`
      ).join("");
    }

    function renderResults(results) {
      document.getElementById("results").innerHTML = (results || []).map(item => `
        <div class="result">
          <strong>${escapeHtml(item.document)} / section ${item.chunk}</strong>
          <span class="muted">score ${item.score}</span>
          <pre>${escapeHtml(item.text)}</pre>
        </div>
      `).join("") || "No matching sources.";
    }

    function renderWarnings(warnings) {
      if (!warnings.length) return;
      document.getElementById("citations").innerHTML = warnings.map(warning =>
        `<span class="citation warn">${escapeHtml(warning)}</span>`
      ).join("");
    }

    function setAnswer(text, muted) {
      const answer = document.getElementById("answer");
      answer.textContent = text;
      answer.classList.toggle("muted", muted);
      document.getElementById("citations").innerHTML = "";
    }

    async function getJson(url, options) {
      const response = await fetch(url, options || {});
      return await response.json();
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[char]));
    }
  </script>
</body>
</html>
"""


class AssistantHandler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = render_home()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/ask":
            question = parse_qs(parsed.query).get("q", [""])[0]
            self.send_json(answer_question(question))
            return
        if parsed.path == "/api/summarize":
            question = parse_qs(parsed.query).get("q", [""])[0]
            self.send_json(summarize(question))
            return
        if parsed.path == "/api/documents":
            self.send_json({"documents": document_status()})
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        if parsed.path == "/api/reindex":
            self.send_json(index_documents())
            return

        if parsed.path == "/api/upload":
            parts = parse_multipart(body, self.headers.get("Content-Type", ""))
            upload = (parts.get("file") or [{}])[0]
            filename = upload.get("filename", "")
            content = upload.get("content", b"")
            if not filename or not content:
                self.send_json({"message": "No file selected."}, 400)
                return
            path = unique_path(DOCUMENTS_DIR, filename)
            path.write_bytes(content)
            result = index_documents()
            self.send_json({"message": f"Uploaded {path.name}.", **result})
            return

        if parsed.path == "/api/note":
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                self.send_json({"message": "Invalid note data."}, 400)
                return
            path = create_note(payload.get("title", ""), payload.get("body", ""))
            result = index_documents()
            self.send_json({"message": f"Saved {path.name}.", **result})
            return

        self.send_json({"error": "Not found"}, 404)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    index_documents()
    server = ThreadingHTTPServer((host, port), AssistantHandler)
    print(f"Document assistant running at http://{host}:{port}")
    server.serve_forever()


def install_help() -> str:
    return (
        "PDF files are supported when pypdf or PyPDF2 is installed. "
        "Run: python -m pip install pypdf"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local AI-style document assistant.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="Create sample documents.")
    sample.add_argument("--folder", type=Path, default=DOCUMENTS_DIR)

    index = subparsers.add_parser("index", help="Index documents.")
    index.add_argument("--folder", type=Path, default=DOCUMENTS_DIR)
    index.add_argument("--db", type=Path, default=DEFAULT_DB)

    ask = subparsers.add_parser("ask", help="Ask a question from the terminal.")
    ask.add_argument("question")
    ask.add_argument("--db", type=Path, default=DEFAULT_DB)

    summary = subparsers.add_parser("summarize", help="Summarize matching sections.")
    summary.add_argument("query", nargs="?", default="")
    summary.add_argument("--db", type=Path, default=DEFAULT_DB)

    note = subparsers.add_parser("note", help="Add a note from the terminal.")
    note.add_argument("title")
    note.add_argument("body")

    upload = subparsers.add_parser("upload", help="Copy a document into the library.")
    upload.add_argument("path", type=Path)

    serve_command = subparsers.add_parser("serve", help="Run the browser app.")
    serve_command.add_argument("--host", default="127.0.0.1")
    serve_command.add_argument("--port", type=int, default=8080)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sample":
        create_sample_documents(args.folder)
        print(f"Sample documents created in: {args.folder}")
    elif args.command == "index":
        result = index_documents(args.folder, args.db)
        print(f"Indexed {result['documents']} document(s) and {result['chunks']} section(s).")
        for warning in result["warnings"]:
            print(f"Warning: {warning}")
    elif args.command == "ask":
        response = answer_question(args.question, args.db)
        print(html.unescape(response["answer"]))
        if response["citations"]:
            print("\nCitations:")
            for citation in response["citations"]:
                print(f"- {citation['document']} / section {citation['chunk']}")
    elif args.command == "summarize":
        response = summarize(args.query, args.db)
        print(response["summary"])
    elif args.command == "note":
        path = create_note(args.title, args.body)
        result = index_documents()
        print(f"Saved {path.name}. Indexed {result['documents']} document(s).")
    elif args.command == "upload":
        if not args.path.exists():
            parser.error(f"File not found: {args.path}")
        destination = unique_path(DOCUMENTS_DIR, args.path.name)
        shutil.copy2(args.path, destination)
        result = index_documents()
        print(f"Uploaded {destination.name}. Indexed {result['documents']} document(s).")
        if destination.suffix.lower() == ".pdf" and result["warnings"]:
            print(install_help())
    elif args.command == "serve":
        serve(args.host, args.port)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
