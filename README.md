# LearnNova AI

LearnNova AI is a local study assistant for PDF notes. It provides a
Streamlit website for asking questions about a document, reviewing a quick
summary, and generating practice material. Ollama runs the language and
embedding models locally, while ChromaDB stores the processed document chunks
on the computer running the app.

## Features

- **PDF library:** Upload multiple PDFs and choose which one is active.
- **PDF reading:** Extracts selectable text page by page with `pypdf`. Pages
  without selectable text can use optional OCR support.
- **Ask Your Notes:** Ask follow-up questions and receive answers grounded in
  retrieved passages. Answers stream into the page as Ollama generates them.
- **PDF Summary:** Build and download a quick extractive summary, including
  key points and definitions. This summary is generated locally from the
  extracted text; it does not call a language model.
- **Exam Preparation Studio:** Generate MCQs, short- and long-answer
  questions, passage-based questions, mixed questions, revision notes, and
  flashcards. These formats are generated from source statements in the PDF.
  MCQs can be attempted and scored in the app.
- **Downloads and sources:** Download summaries and study material as
  Markdown, and inspect the passages used for an answer or generated material.

## Requirements

- Windows, macOS, or Linux
- Python and `pip`
- [Ollama](https://ollama.com/) installed and running locally
- The Ollama models used by the app:
  - `qwen2.5:3b` for chat answers
  - `nomic-embed-text` for initial PDF indexing
- A PDF with selectable text. Scanned/image-only PDFs require the optional
  OCR setup below.

The app's dependencies are listed in [`requirements.txt`](requirements.txt).

## Install and run on Windows

Open PowerShell in the project folder:

```powershell
cd "C:\path\to\StudyMateAI"
```

Create and activate a virtual environment, then install the Python packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, either allow local scripts for
the current user or invoke the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install and start Ollama, then download the required models:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Ollama must be running while you use the app. If it is not already running as
a background service, start it in a separate terminal:

```powershell
ollama serve
```

In the project terminal, start LearnNova AI:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your
browser, upload a PDF, and select it as the active document.

## Optional OCR for scanned PDFs

OCR is optional; ordinary text-based PDFs do not need it. The app attempts to
enable OCR when `PyMuPDF`, `pytesseract`, and Pillow are importable and
Tesseract OCR is installed. Install the Python packages in the active virtual
environment:

```powershell
python -m pip install PyMuPDF pytesseract Pillow
```

Install Tesseract OCR separately. The current app configuration expects
`C:\Program Files\Tesseract-OCR\tesseract.exe`. If Tesseract is installed
elsewhere, update `pytesseract.pytesseract.tesseract_cmd` near the top of
[`app.py`](app.py). OCR may take longer than extracting selectable text, and
low-quality scans may not be recognized accurately.

## Using the app

1. Start Ollama and run the Streamlit command above.
2. Upload a PDF using **Upload your study notes (PDF)**.
3. Wait for the app to extract the text and prepare the document index. This
   initial processing may take time, depending on the PDF size and computer.
4. Choose the active PDF if you have uploaded more than one.
5. Open one of the tabs:
   - **Ask Your Notes:** Enter a question in the chat box. Short follow-up
     questions can refer to recent conversation history.
   - **PDF Summary:** Generate or download the extractive summary.
   - **Exam Preparation Studio:** Select an optional topic, format, difficulty,
     answer style, marks where applicable, and number of items, then generate.
     For MCQs, select answers and submit to view your score.

## How it works

### PDF processing and storage

The active app is [`app.py`](app.py). It reads PDF pages with `pypdf`, cleans
the extracted text, and splits it into overlapping word chunks. During initial
indexing, the chunks are embedded with Ollama's `nomic-embed-text` model and
stored in a persistent ChromaDB database. Collections are associated with a
document hash and embedding-version suffix so different PDFs and index
versions do not share chunks.

The database is created in `studymate_db` relative to the directory from
which Streamlit is launched. Keep this folder if you want to retain the local
index between runs. It can be rebuilt by removing/recreating the database,
but do so only when you intend to discard the stored indexes.

### Question answering

The app ranks cached document chunks locally against the words in the
question, then sends a small set of matching passages and recent conversation
history to the configured Ollama chat model. The response is streamed to the
browser. The first answer may be slower while Ollama loads the model; later
answers are typically faster while it remains loaded.

### Summaries and exam material

The PDF summary is extractive: it selects and organizes sentences from the
source text rather than asking an AI model to write a new summary. Exam
material is also generated locally from readable statements found in the PDF.
This keeps these features fast and limits unsupported claims, but it can
produce fewer items than requested when the topic has too few distinct
statements. The app reports this rather than intentionally filling gaps with
invented material.

## Project files

| File or folder | Purpose |
| --- | --- |
| `app.py` | Main Streamlit application. Run this file to launch the website. |
| `requirements.txt` | Python dependencies for the main application. |
| `studymate_db/` | Persistent ChromaDB indexes created at runtime. |
| `LearnNova_AI_app.py` | Earlier alternate Streamlit app; `app.py` is the current entrypoint. |
| `read_pdf.py` | Small command-line PDF text extraction experiment. |
| `chunk_pdf.py` | Command-line PDF chunking experiment. |
| `embedding_test.py` | Tests generating an Ollama embedding from a sentence. |
| `embed_pdf.py` | Experiment embedding PDF chunks with Ollama. |
| `search_pdf.py` | Experiment retrieving PDF chunks with cosine similarity. |
| `store_pdf.py` | Experiment storing PDF chunks and embeddings in ChromaDB. |
| `search_chroma.py` | Experiment querying a ChromaDB collection. |
| `rag_chat.py` | Command-line retrieval-augmented chat prototype. |
| `rag_chroma.py` | Command-line chat prototype using a ChromaDB collection. |
| `app_backup.py`, `app.zip` | Backup/archive files; not required to run the current app. |

The command-line experiments are separate from the website and may have
different setup requirements. In particular, some use PyMuPDF directly.

## Configuration

The model names and ChromaDB location are set near the top of `app.py`:

```python
CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
client = chromadb.PersistentClient(path="./studymate_db")
```

If you change a model name, pull that model with Ollama before running the
app. The embedding model is needed when indexing a new PDF; the chat model is
needed for question answering.

## Troubleshooting

### Ollama connection error

- Confirm Ollama is installed and running.
- In a terminal, run `ollama list` and verify both required models appear.
- If Ollama is not running as a service, start `ollama serve` in a separate
  terminal and retry.

### Model not found

Run:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### PDF has no readable text

Some PDFs contain page images rather than selectable text. Enable the
[optional OCR setup](#optional-ocr-for-scanned-pdfs), or use a text-based PDF.
The app also shows extraction status and a preview when text is available.

### Processing or answers are slow

The first indexing run embeds every chunk and is usually the slowest. Later
runs reuse the persistent index. Chat speed depends on the computer and the
time Ollama takes to load and run the configured model. Closing other
GPU/CPU-intensive applications or using a smaller Ollama chat model may help;
if you change the model, update `CHAT_MODEL` in `app.py` and pull the matching
model first.

### PowerShell will not activate `.venv`

Use the virtual environment's Python executable directly as shown in the
installation steps, or adjust the PowerShell execution policy according to
your organization's requirements.

## Privacy and limitations

- PDF extraction, ChromaDB indexing, and Ollama inference are configured to
  run locally. The app does not intentionally upload documents to a hosted
  AI service.
- The PDF index is persisted locally in `studymate_db`; protect or remove
  that folder if the indexed material is sensitive.
- Answers and generated study material depend on the quality and completeness
  of the extracted PDF text. Always check important facts against the
  original document.
- OCR is optional and may miss text in poor-quality scans, diagrams, tables,
  or complex layouts.
- This project is a study aid, not a substitute for checking course
  materials or instructor guidance.
