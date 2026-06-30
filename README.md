# AI Document Assistant

A local Python app for uploading documents or notes, searching them with a hybrid semantic score, asking cited questions, and summarizing matching sections.

## What It Can Do

- Upload `.pdf`, `.txt`, `.md`, `.csv`, `.json`, `.html`, and `.htm` files
- Save pasted notes directly from the browser
- Reindex your document library
- Search across indexed sections
- Ask questions and get cited source snippets
- Summarize the whole library or a focused topic

## Run The App

```bash
python doc_assistant.py serve
```

Then open:

```text
http://127.0.0.1:8080
```

## PDF Support

PDF upload is built in. Text extraction from PDFs needs one optional library:

```bash
python -m pip install pypdf
```

If `pypdf` or `PyPDF2` is not installed, the app still saves uploaded PDFs and shows a clear warning that text extraction is not available yet.

## Command-Line Use

Create sample documents:

```bash
python doc_assistant.py sample
```

Index documents:

```bash
python doc_assistant.py index
```

Upload a file:

```bash
python doc_assistant.py upload path/to/document.pdf
```

Add a note:

```bash
python doc_assistant.py note "Meeting Notes" "Budget review happens every Friday."
```

Ask a question:

```bash
python doc_assistant.py ask "How much is the home office reimbursement?"
```

Summarize matching sections:

```bash
python doc_assistant.py summarize "remote work reimbursement"
```

## How It Works

This version runs locally and does not call a paid AI API. It uses a hybrid local ranking method: TF-IDF keyword matching, phrase overlap, and short word-sequence similarity. That gives practical semantic-style retrieval while keeping the project simple to run.

A later upgrade can add real embedding models, OCR for scanned PDFs, chat history, and OpenAI-powered answer generation.
