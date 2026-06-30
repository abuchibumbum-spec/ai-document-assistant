# AI Document Assistant

A local, dependency-free document assistant built in pure Python. Upload files or notes, index them automatically, search with a hybrid keyword + semantic scoring approach, ask questions and get cited answers, or generate summaries — all through a built-in web interface, no external services or API keys required.

## Features

- **Multi-format support** — indexes `.txt`, `.md`, `.csv`, `.json`, `.html`, and `.pdf` files (PDF text extraction requires `pypdf` or `PyPDF2`)
- **Hybrid search** — combines TF-IDF style keyword scoring with shingle-based semantic overlap for more relevant matches
- **Cited answers** — questions are answered with extracted sentences and source citations pointing back to the originating document and section
- **Summarization** — generates extractive summaries of indexed content, optionally focused on a query
- **Notes** — add free-text notes directly, which get indexed alongside uploaded documents
- **Built-in web UI** — runs a local HTTP server with upload, search, and Q&A, no frontend framework needed
- **Zero external dependencies** — built entirely with the Python standard library (PDF support is optional and only needs `pypdf`)

## Getting Started

### Requirements
- Python 3.10+
- Optional: `pypdf` for PDF text extraction (`pip install pypdf`)

### Installation
```bash
git clone https://github.com/abuchibumbum-spec/ai-document-assistant.git
cd ai-document-assistant
```

## Usage

### Run the web app (recommended)
```bash
python document_assistant.py serve
```
Open `http://127.0.0.1:8080` in your browser to upload files, add notes, search, and ask questions.

### Or use the command line

**Generate sample documents**
```bash
python document_assistant.py sample
```

**Index documents** (after adding files to the `documents/` folder)
```bash
python document_assistant.py index
```

**Ask a question**
```bash
python document_assistant.py ask "How many vacation days do employees get?"
```

**Summarize content**
```bash
python document_assistant.py summarize "expense policy"
```

**Add a note**
```bash
python document_assistant.py note "Meeting Notes" "Discussed Q3 roadmap and budget."
```

**Upload an existing file into the library**
```bash
python document_assistant.py upload path/to/file.pdf
```

## How It Works

1. Documents are split into overlapping chunks (~220 words, 55-word overlap) for better retrieval granularity.
2. Each chunk is scored against a query using a blend of TF-IDF keyword matching and shingle-based phrase overlap.
3. For questions, the top-matching chunks are broken into sentences, ranked by relevance, and combined into a cited answer.
4. For summaries, an extractive scoring method picks the most representative sentences from matching content.

## License

Feel free to use and adapt this project.
