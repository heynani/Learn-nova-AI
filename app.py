import streamlit as st
import ollama
import chromadb
import pypdf
import hashlib
import html
import re
import io
from collections import Counter

# Optional OCR support for scanned PDFs.
# If these imports fail, the app still works normally for text PDFs.
try:
    import pymupdf
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    from PIL import Image
    OCR_AVAILABLE = True
    OCR_ERROR = ""
except Exception as error:
    OCR_AVAILABLE = False
    OCR_ERROR = str(error)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="LearnNova AI",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245, 247, 251, .92); }
    [data-testid="stSidebar"] { background: #eef2fa; }
    .block-container { max-width: 1260px; padding-top: 1.8rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #17233f; }
    .ln-hero {
        padding: 30px 32px; border-radius: 22px; margin: 4px 0 22px;
        background: linear-gradient(120deg, #18294f 0%, #334f91 58%, #6c69b7 100%);
        color: white; box-shadow: 0 14px 34px rgba(35, 55, 105, .16);
    }
    .ln-hero h1 { color: white; font-size: 2.15rem; margin: 0 0 8px; }
    .ln-hero p { color: #e4eaff; font-size: 1.02rem; margin: 0; }
    .ln-kicker {
        color: #c9d6ff; font-size: .78rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: 9px;
    }
    .ln-card {
        background: white; border: 1px solid #e6eaf2; border-radius: 17px;
        padding: 17px 18px; box-shadow: 0 5px 18px rgba(25, 42, 78, .04);
    }
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e6eaf2; padding: 15px 17px;
        border-radius: 15px; box-shadow: 0 4px 15px rgba(25, 42, 78, .035);
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 11px; font-weight: 600; min-height: 2.65rem;
    }
    div[data-testid="stTabs"] button { font-weight: 650; }
    div[data-testid="stAlert"] { border-radius: 13px; }
    .ln-small { color: #69758d; font-size: .9rem; }
</style>
<div class="ln-hero">
  <div class="ln-kicker">Your personal learning companion</div>
  <h1>📚 LearnNova AI</h1>
  <p>Turn your study PDFs into clear summaries, helpful answers and smarter practice.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✨ LearnNova AI")
    st.caption("Your personal AI study space")
    st.markdown("---")
    st.markdown("**Your study workflow**")
    st.markdown("1. Upload your PDF notes")
    st.markdown("2. Read the quick PDF summary")
    st.markdown("3. Ask questions from your notes")
    st.markdown("4. Practise with exam materials")
    st.markdown("---")
    st.caption("Powered locally by Ollama + ChromaDB")

# ==========================================
# 2. SETTINGS, DATABASE CONNECTION, EMBEDDING HELPERS
# ==========================================
CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"
SUMMARY_VERSION = 2
TOPIC_SUGGESTION_VERSION = 2

# nomic-embed-text expects task prefixes ("search_document:", "search_query:").
# Bumping this version rebuilds collections when the PDF extraction pipeline
# changes, preventing stale chunks from bypassing the updated reader.
EMBEDDING_VERSION = "v3"

client = chromadb.PersistentClient(path="./studymate_db")


def embed_texts(texts):
    """Embed a batch of document chunks with the document prefix."""
    inputs = [f"search_document: {text}" for text in texts]
    result = ollama.embed(model=EMBED_MODEL, input=inputs)
    return result["embeddings"]


def embed_query(text):
    """Embed a student question or topic with the query prefix."""
    result = ollama.embed(model=EMBED_MODEL, input=f"search_query: {text}")
    return result["embeddings"][0]


# ==========================================
# 3. DOCUMENT LIBRARY (MULTIPLE PDFs)
# ==========================================
if "library" not in st.session_state:
    st.session_state.library = {}
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

uploaded_file = st.file_uploader(
    "Upload your study notes (PDF)",
    type=["pdf"]
)

uploaded_hash = None
if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    uploaded_hash = hashlib.sha256(pdf_bytes).hexdigest()[:20]
    if uploaded_hash != st.session_state.last_uploaded:
        st.session_state.library[uploaded_hash] = uploaded_file.name
        st.session_state.last_uploaded = uploaded_hash
        st.session_state["active_doc_selector"] = uploaded_hash

if not st.session_state.library:
    st.info("Upload a PDF to start chatting or prepare for exams.")
    st.stop()

document_hashes = list(st.session_state.library.keys())
if (
    "active_doc_selector" in st.session_state
    and st.session_state["active_doc_selector"] not in document_hashes
):
    st.session_state.pop("active_doc_selector")

default_index = 0
if st.session_state.last_uploaded in document_hashes:
    default_index = document_hashes.index(st.session_state.last_uploaded)

file_hash = st.selectbox(
    "Active document",
    options=document_hashes,
    index=default_index,
    format_func=lambda h: st.session_state.library[h],
    key="active_doc_selector"
)

collection_name = f"notes_{file_hash}_{EMBEDDING_VERSION}"

collection = client.get_or_create_collection(
    name=collection_name,
    metadata={
        "hnsw:space": "cosine",
        "description": "StudyMate uploaded PDF notes"
    }
)

# ==========================================
# 4. EXTRACT TEXT AND STORE EMBEDDINGS (WITH OCR FALLBACK)
# ==========================================
def normalize_pdf_text(text):
    """Clean extracted line breaks without joining unrelated words."""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-[ \t]*\n[ \t]*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_bytes):
    """Extract text page-by-page and OCR pages with no selectable text."""
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    page_texts = []
    extraction_errors = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text(extraction_mode="layout") or ""
        except Exception as error:
            page_text = ""
            extraction_errors.append((page_number, str(error)))
        page_texts.append(normalize_pdf_text(page_text))

    missing_pages = [
        page_number
        for page_number, page_text in enumerate(page_texts, start=1)
        if not page_text
    ]
    ocr_errors = []

    if missing_pages and OCR_AVAILABLE:
        document = None
        try:
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            for page_number in missing_pages:
                try:
                    page = document.load_page(page_number - 1)
                    pix = page.get_pixmap(dpi=250)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_texts[page_number - 1] = normalize_pdf_text(
                        pytesseract.image_to_string(image)
                    )
                except Exception as error:
                    ocr_errors.append(f"Page {page_number}: {error}")
        except Exception as error:
            ocr_errors.append(f"Could not open PDF for OCR: {error}")
        finally:
            if document is not None:
                document.close()

    pages_without_text = [
        page_number
        for page_number, page_text in enumerate(page_texts, start=1)
        if not page_text
    ]
    extraction_errors = [
        f"Page {page_number}: {error}"
        for page_number, error in extraction_errors
        if page_number in pages_without_text
    ]
    full_text = "\n\n".join(
        f"Page {page_number}\n{page_text}"
        for page_number, page_text in enumerate(page_texts, start=1)
        if page_text
    )
    return {
        "text": full_text,
        "page_count": len(reader.pages),
        "pages_without_text": pages_without_text,
        "extraction_errors": extraction_errors,
        "ocr_errors": ocr_errors,
    }


if collection.count() == 0:
    if uploaded_hash != file_hash:
        st.warning(
            "This document has not been processed yet. Select it in the "
            "uploader above (upload the PDF file once) so Learnnova AI can "
            "read and store it."
        )
        st.stop()

    full_text = ""
    extraction_error = None

    with st.spinner("Reading your PDF and preparing notes..."):
        try:
            pdf_report = extract_pdf_text(pdf_bytes)
            full_text = pdf_report["text"]
            st.session_state["pdf_read_report"] = {
                **pdf_report,
                "document": file_hash,
            }
        except Exception as error:
            extraction_error = str(error)

    if extraction_error:
        st.error(f"Could not process this PDF: {extraction_error}")
        st.stop()

    if not full_text:
        if OCR_AVAILABLE:
            st.error(
                "No text could be extracted from this PDF, even after OCR. "
                "Check that the pages are readable and the PDF is not damaged."
            )
        else:
            st.error(
                "No selectable text was found in this PDF. To read scanned "
                "PDFs, PyMuPDF, Tesseract OCR, and pytesseract are required. "
                f"OCR is unavailable: {OCR_ERROR}"
            )
        st.stop()

    if pdf_report["pages_without_text"]:
        pages = ", ".join(str(page) for page in pdf_report["pages_without_text"])
        if pdf_report["ocr_errors"]:
            ocr_status = "OCR errors: " + "; ".join(pdf_report["ocr_errors"])
        elif not OCR_AVAILABLE:
            ocr_status = f"OCR is unavailable: {OCR_ERROR}"
        else:
            ocr_status = "Check that the scanned pages are readable."
        st.warning(
            f"No text could be extracted from PDF page(s): {pages}. "
            "Those pages will not be included in chat or summaries. "
            + ocr_status
        )
    if pdf_report["extraction_errors"]:
        st.warning(
            "Some pages could not be read: "
            + "; ".join(pdf_report["extraction_errors"])
        )

    with st.spinner("Splitting and embedding your notes..."):
        try:
            words = full_text.split()
            chunk_size = 250
            overlap = 40
            chunks = []
            start = 0

            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunks.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start = end - overlap

            batch_size = 32
            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start:batch_start + batch_size]
                embeddings = embed_texts(batch)
                ids = [
                    f"chunk_{i}"
                    for i in range(batch_start, batch_start + len(batch))
                ]
                collection.add(
                    ids=ids,
                    documents=batch,
                    embeddings=embeddings,
                    metadatas=[
                        {"chunk_index": batch_start + i}
                        for i in range(len(batch))
                    ]
                )
        except Exception as error:
            client.delete_collection(name=collection_name)
            st.error(
                "Could not finish indexing this PDF. The partial index was "
                f"removed so it can be retried safely. Details: {error}"
            )
            st.stop()

    st.success(f"Prepared {len(chunks)} chunks.")

st.caption(f"📄 Document: {st.session_state.library[file_hash]}")
st.caption(f"📚 Stored chunks: {collection.count()}")

pdf_read_report = st.session_state.get("pdf_read_report")
if pdf_read_report and pdf_read_report.get("document") == file_hash:
    readable_pages = (
        pdf_read_report["page_count"]
        - len(pdf_read_report["pages_without_text"])
    )
    st.caption(
        f"Selectable text found on {readable_pages} of "
        f"{pdf_read_report['page_count']} pages. Text embedded only in "
        "page images may require OCR."
    )
    with st.expander("Preview extracted PDF text"):
        st.text(pdf_read_report["text"][:3000])

if collection.count() == 0:
    st.warning("No text chunks are available in this document.")
    st.stop()

overview_a, overview_b, overview_c = st.columns(3)
with overview_a:
    st.metric(
        "Study document",
        "Ready",
        help="Your selected PDF has been processed."
    )
with overview_b:
    st.metric("Knowledge chunks", collection.count())
with overview_c:
    st.metric(
        "Learning tools",
        "3",
        help="PDF summary, notes chat and exam practice."
    )
st.markdown(
    f'<div class="ln-card"><strong>📄 Currently studying:</strong> '
    f'{html.escape(st.session_state.library[file_hash])} '
    '<span class="ln-small"> · Your learning workspace is ready</span></div>',
    unsafe_allow_html=True
)


# ==========================================
# 5. RESET PER-DOCUMENT STATE WHEN THE PDF CHANGES
# ==========================================
if st.session_state.get("active_document") != file_hash:
    st.session_state.messages = []
    for state_key in [
        "exam_material",
        "exam_document",
        "exam_sources",
        "exam_distances",
        "exam_format",
        "exam_format_valid",
        "exam_format_warning",
        "flashcard_pairs",
        "quiz_questions",
        "quiz_submitted",
        "document_summary",
        "summary_document",
        "summary_version"
    ]:
        st.session_state.pop(state_key, None)
    st.session_state.active_document = file_hash


# ==========================================
# 6. SHARED RETRIEVAL, COVERAGE, DISPLAY, AND CHAT FUNCTIONS
# ==========================================
def retrieve_context(search_text, number_of_chunks=12):
    """Retrieve passages for the student's topic/question, re-ranked so
    passages that actually mention the topic come first."""

    # Guard: querying an empty collection can raise an error
    if collection.count() == 0:
        return [], [], []

    query_vector = embed_query(search_text)

    # Pull a wider candidate pool, then re-rank for topic focus
    candidate_count = min(number_of_chunks * 3, collection.count())
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_count,
        include=["documents", "distances", "metadatas"]
    )

    passages = results["documents"][0] or []
    distances = results["distances"][0] or []
    metadatas = results["metadatas"][0] or []

    # Meaningful words from what the student typed
    topic_terms = {
        word for word in re.findall(r"[a-z0-9]+", search_text.lower())
        if len(word) >= 3
    }

    def topic_key(passage, distance):
        words = re.findall(r"[a-z0-9]+", passage.lower())
        word_set = set(words)
        distinct_hits = len(topic_terms & word_set)
        total_hits = sum(1 for word in words if word in topic_terms)
        return (-distinct_hits, -total_hits, distance)

    # Soft re-rank: nothing is deleted, so this can never return zero results.
    ranked = sorted(
        zip(passages, distances, metadatas),
        key=lambda item: topic_key(item[0], item[1])
    )[:number_of_chunks]

    passages = [item[0] for item in ranked]
    distances = [item[1] for item in ranked]
    metadatas = [item[2] for item in ranked]

    return passages, distances, metadatas


def topic_coverage_message(topic, passages):
    """Warn when the selected topic barely appears in the retrieved passages."""
    topic = (topic or "").strip()
    if not topic or not passages:
        return None
    topic_terms = {
        word for word in re.findall(r"[a-z0-9]+", topic.lower())
        if len(word) >= 3
    }
    if not topic_terms:
        return None
    best_hits = 0
    for passage in passages:
        passage_words = set(re.findall(r"[a-z0-9]+", passage.lower()))
        best_hits = max(best_hits, len(topic_terms & passage_words))
    if best_hits == 0:
        return (
            f'The retrieved notes barely mention "{topic}". The material '
            "below should state which parts the document does not cover. "
            "If the results look unrelated, check the topic spelling or "
            "make sure the correct PDF is uploaded."
        )
    return None


def make_context(passages):
    """Label passages so the model can refer to evidence clearly."""
    return "\n\n".join(
        f"Source passage {i + 1}:\n{passage}"
        for i, passage in enumerate(passages)
    )


def get_all_document_chunks():
    """Return every stored chunk in original document order."""
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    paired = list(zip(documents, metadatas))
    paired.sort(key=lambda item: item[1].get("chunk_index", 0))
    return [document for document, _ in paired]


def extract_definition_facts(chunks):
    """Extract short, source-backed definitions from the PDF text."""
    text = "\n".join(chunks)
    pattern = re.compile(
        r"^([A-Z][A-Za-z0-9][A-Za-z0-9 /()&'-]{1,55}?)\s+"
        r"(?i:is defined as|is a|is an|is the|are|refers to|means|denotes|"
        r"consists of|is called|is a type of)\s+"
        r"(.{20,240})$",
    )
    excluded_subjects = {
        "a", "an", "after", "because", "before", "how", "if", "it",
        "these", "this", "those", "they", "we", "what", "when", "where",
        "which", "who", "why", "you", "suppose", "for", "here", "thus",
        "since", "o", "task", "following", "based", "therefore"
    }
    facts = []
    seen = set()
    for raw_sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = re.sub(r"\s+", " ", raw_sentence).strip(" -*\ufffd")
        sentence = re.sub(
            r"^.*?\b(?:formal\s+)?definition\s*:\s*",
            "",
            sentence,
            flags=re.IGNORECASE
        )
        sentence = re.sub(
            r",?\s+also known as\s+[^,]+,?",
            "",
            sentence,
            flags=re.IGNORECASE
        )
        match = pattern.match(sentence)
        if not match:
            continue
        subject = re.sub(
            r"^[^A-Za-z0-9]+|[^A-Za-z0-9)]+$",
            "",
            re.sub(r"\s+", " ", match.group(1)).strip(" -:")
        )
        subject = re.sub(
            r"^(.{3,40}?)\s+\1$",
            r"\1",
            subject,
            flags=re.IGNORECASE
        )
        subject = re.sub(r"^(?:a|an|the)\s+", "", subject, flags=re.IGNORECASE)
        definition = re.sub(r"\s+", " ", match.group(2)).strip(" -:;")
        subject_words = set(re.findall(r"[a-z]+", subject.casefold()))
        key = subject.casefold()
        if (
            subject.split()[0].casefold() in excluded_subjects
            or len(subject) < 3
            or len(subject.split()) > 5
            or ":" in subject
            or "," in subject
            or subject_words & excluded_subjects
            or len(definition.split()) < 4
            or re.search(
                r"\b(is|are|was|were)\s+\w+\b.*\b\1\s+\w+\b",
                definition,
                flags=re.IGNORECASE
            )
            or key in seen
        ):
            continue
        seen.add(key)
        facts.append({"subject": subject, "definition": definition})
    return facts


def build_document_summary(chunks, max_points=12):
    """Build a fast extractive summary without a model round trip."""
    sentences = []
    seen = set()
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk):
            sentence = re.sub(r"\s+", " ", sentence).strip(" \t-*")
            sentence = re.sub(
                r"\b[A-Z][A-Z. ]{5,40},\s*[A-Z. ]{3,30},\s*"
                r"REVA UNIVERSITY\s+Page\s+\d+\s*",
                "",
                sentence,
                flags=re.IGNORECASE
            )
            sentence = re.sub(r"\bPage\s+\d+\s*", "", sentence, flags=re.IGNORECASE)
            words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", sentence)
            normalized = sentence.casefold()
            repeated_subject = re.search(
                r"\b([A-Za-z]+)\s+(is|are|was|were)\s+\w+.{1,80}"
                r"\b\1\s+\2\b",
                sentence,
                flags=re.IGNORECASE
            )
            if repeated_subject:
                continue
            if (
                len(words) < 8
                or len(words) > 55
                or not sentence[0].isupper()
                or sentence[-1] not in ".!?"
                or re.search(r":\s*\d+\.$", sentence)
                or re.search(
                    r"\b(?:a|an|the|to|of|for|with|and|or|is|are|as|"
                    r"in|on|by|which|into|from|because|using)\s*[.!?]$",
                    sentence,
                    flags=re.IGNORECASE
                )
                or normalized in seen
                or sentence.lower().startswith(("page ", "unit ", "chapter "))
            ):
                continue
            seen.add(normalized)
            sentences.append((sentence, words))

    if not sentences:
        fallback_points = []
        for chunk in chunks:
            for line in chunk.splitlines():
                point = re.sub(r"\s+", " ", line).strip(" -*")
                if point and point.casefold() not in {
                    existing.casefold() for existing in fallback_points
                }:
                    fallback_points.append(point[:500])
                if len(fallback_points) >= max_points:
                    break
            if len(fallback_points) >= max_points:
                break
        if not fallback_points:
            raise ValueError("No readable text was found in this PDF.")
        lines = [
            "## Overview",
            "",
            fallback_points[0],
            "",
            "## Key points"
        ]
        lines.extend(f"- {point}" for point in fallback_points[1:])
        return "\n".join(lines)

    stop_words = {
        "about", "after", "also", "and", "are", "because", "been", "before",
        "being", "between", "both", "can", "could", "does", "during", "each",
        "for", "from", "further", "had", "has", "have", "into", "its", "may",
        "more", "most", "not", "of", "often", "only", "other", "our", "over",
        "such", "than", "that", "the", "their", "there", "these", "this",
        "those", "through", "under", "use", "used", "using", "was", "were",
        "which", "while", "with", "would"
    }
    frequencies = Counter(
        word.casefold()
        for _, words in sentences
        for word in words
        if word.casefold() not in stop_words
    )

    ranked = []
    for index, (sentence, words) in enumerate(sentences):
        terms = [
            word.casefold() for word in words
            if word.casefold() not in stop_words
        ]
        if not terms:
            continue
        score = sum(frequencies[word] for word in set(terms)) / len(terms)
        score /= 1 + index / max(len(sentences), 1)
        ranked.append((score, index, sentence, set(terms)))

    selected = []
    selected_terms = set()
    for score, index, sentence, terms in sorted(ranked, reverse=True):
        if selected and len(terms & selected_terms) / max(len(terms), 1) > 0.65:
            continue
        selected.append((index, sentence, terms))
        selected_terms.update(terms)
        if len(selected) >= max_points:
            break
    selected.sort(key=lambda item: item[0])

    definitions = extract_definition_facts(chunks)
    lines = ["## Overview", "", selected[0][1], "", "## Key points"]
    lines.extend(f"- {sentence}" for _, sentence, _ in selected[1:])
    if definitions:
        lines.extend(["", "## Important definitions"])
        lines.extend(
            f"- **{fact['subject']}**: {fact['definition']}."
            for fact in definitions[:6]
        )
    lines.extend([
        "",
        "> Quick, source-extracted revision notes. Review the original PDF "
        "for full context."
    ])
    return "\n".join(lines)


def generate_document_summary():
    """Create a quick, factual study summary directly from stored PDF text."""
    chunks = get_all_document_chunks()
    if not chunks:
        raise ValueError("No readable text chunks were found in this PDF.")
    return build_document_summary(chunks)


def format_for_display(text):
    """Fix model output so it renders correctly in Markdown.

    Pass 0:  rejoin a bold label the model wrapped across two lines
             ('**Correct' + 'answer:** ...' -> '**Correct answer:** ...').
    Pass 0b: split a bold label glued mid-line onto its own line
             ('**Correct answer:** B. x **Explanation:** y' -> two lines).
    Pass 1:  split the first option off a question line when the model
             wrote them together ('**Question?** A. option').
    Pass 2:  add Markdown hard line-breaks so question lines, options,
             answers, and explanations never merge into one paragraph.
    """
    lines = text.splitlines()

    # ---------- Pass 0: rejoin wrapped bold labels ----------
    label_word = r"(?:Correct|Model|Answer|Explanation)"
    head_pattern = re.compile(
        r"^\s*(?:[-*•]\s*)?\*\*\s*" + label_word + r"(?:\s+answer)?\s*$",
        re.IGNORECASE
    )
    tail_pattern = re.compile(
        r"^\s*(?:\*\*)?\s*(?:(?:answer)?\s*:|\*\*)",
        re.IGNORECASE
    )

    rejoined = []
    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else None
        if (
            next_line is not None
            and head_pattern.match(line)
            and tail_pattern.match(next_line)
        ):
            separator = (
                "" if next_line.lstrip().startswith((":", "**")) else " "
            )
            rejoined.append(line.rstrip() + separator + next_line.strip())
            i += 2  # both lines consumed
        else:
            rejoined.append(line)
            i += 1
    lines = rejoined

    # ---------- Pass 0b: split inline bold labels onto their own lines ----------
    inline_label = re.compile(
        r"\s+(?=\*\*\s*(?:Correct answer|Model answer|Explanation)"
        r"(?:\s*:\s*\*\*|\s*\*\*\s*:))",
        re.IGNORECASE
    )
    expanded = []
    for line in lines:
        expanded.extend(inline_label.split(line))
    lines = expanded

    # ---------- Pass 1: split question line if option A is glued on ----------
    answer_label = re.compile(
        r"^\s*(?:[-*•]\s*)?(?:\*\*)?"
        r"(?:Correct answer|Model answer|Answer|Explanation)\s*:",
        re.IGNORECASE
    )
    glued_option = re.compile(
        r"^(.+[?.):*!])\s*((?:\*\*)?A[.):]\s.*)$"
    )

    split_lines = []
    for line in lines:
        if answer_label.match(line):
            split_lines.append(line)
            continue
        match = glued_option.match(line)
        if match:
            split_lines.append(match.group(1).rstrip())
            split_lines.append(match.group(2))
        else:
            split_lines.append(line)

    # ---------- Pass 2: hard line-breaks around options and labels ----------
    option_or_label = re.compile(
        r"^\s*(?:[-*•]\s*)?(?:\*\*)?"
        r"(?:[A-D][.):]|(?:Correct answer|Model answer|Answer|Explanation)\s*:)",
        re.IGNORECASE
    )

    result = []
    for i, line in enumerate(split_lines):
        next_line = split_lines[i + 1] if i + 1 < len(split_lines) else ""
        needs_break = (
            bool(option_or_label.match(line))
            or (bool(option_or_label.match(next_line)) and line.strip() != "")
        )
        result.append(line.rstrip() + "  " if needs_break else line)
    return "\n".join(result)


def generate_chat_answer(question, context, history=None):
    """Generate a structured, grounded answer; history enables follow-ups."""
    history = history or []
    system_prompt = """
You are StudyMate AI, a careful academic tutor helping a student study from
their uploaded notes.

How to answer:
1. Answer the exact question asked, using the conversation so far to resolve
   words like "it", "that", or "more".
2. Start with a direct answer or definition in one or two sentences.
3. Then add a short "Key points" section with 3-6 concise bullet points.
4. Explain every technical term in simple English the first time it appears.
5. Use headings or bullet points so the answer is easy to revise from.
6. Use only the supplied source passages for facts about the notes. Give
   examples only when the passages support them, and never label one example
   as two different concepts.
7. If the passages do not cover part of the question, say exactly which part
   is not covered. Do not fill gaps with outside knowledge while implying it
   came from the notes.
8. Stay focused: do not summarize the whole document when the student asked
   about one specific thing.
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": (
            f"Student question:\n{question}\n\n"
            f"Source passages:\n{context}\n\n"
            "Answer this question using these passages. If something is not "
            "covered by them, say so instead of inventing it."
        )
    })
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        options={"temperature": 0, "num_ctx": 8192}
    )
    return response["message"]["content"].strip()


# ==========================================
# 7. NEW STUDENT FEATURE HELPERS (TOPIC PICKER, FLASHCARDS, QUIZ)
# ==========================================
def get_topic_sample(max_chunks=8):
    """Return a few chunks spread across the stored document."""
    documents = get_all_document_chunks()
    if not documents:
        return []
    if len(documents) <= max_chunks:
        return documents
    step = len(documents) / max_chunks
    return [
        documents[min(int(i * step), len(documents) - 1)]
        for i in range(max_chunks)
    ]


def suggest_topics(passages, limit=8):
    """Suggest concepts that have a definition to support quiz questions."""
    topics = []

    def add_topic(value):
        name = re.sub(r"\s+", " ", value).strip(" .,:;[]")
        if (
            3 <= len(name) <= 48
            and len(name.split()) <= 5
            and name[0].isupper()
            and ":" not in name
            and "," not in name
            and name.casefold() not in {topic.casefold() for topic in topics}
        ):
            topics.append(name)

    for fact in extract_definition_facts(passages):
        add_topic(fact["subject"])

    return topics[:limit]


def study_topic_terms(text):
    """Normalize topic words so plural and verb forms match PDF sentences."""
    ignored = {
        "about", "according", "and", "are", "for", "from", "into", "notes",
        "the", "their", "these", "this", "those", "types", "using", "what"
    }
    terms = set()
    for word in re.findall(r"[a-z]+", text.casefold()):
        if len(word) < 3 or word in ignored:
            continue
        if word.endswith("ies") and len(word) > 5:
            word = word[:-3] + "y"
        elif word.endswith("ing") and len(word) > 6:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 5:
            word = word[:-2]
        elif word.endswith("s") and len(word) > 4:
            word = word[:-1]
        terms.add(word)
    return terms


def extract_study_points(chunks, topic=""):
    """Extract unique, complete, topic-matched statements from PDF chunks."""
    points = []
    seen = set()
    topic_terms = study_topic_terms(topic)
    for chunk in chunks:
        for raw_sentence in re.split(r"(?<=[.!?])\s+|\n+", chunk):
            sentence = re.sub(r"\s+", " ", raw_sentence).strip(" -*")
            sentence = sentence.replace("\ufffd", "")
            sentence = re.sub(r"\bPage\s+\d+\s*", "", sentence, flags=re.IGNORECASE)
            sentence = re.sub(
                r"^.*?\bDefinition\s*:\s*",
                "",
                sentence,
                flags=re.IGNORECASE
            )
            sentence = re.sub(
                r"^.*?\bFormal Definition\b[^:]*:\s*",
                "",
                sentence,
                flags=re.IGNORECASE
            )
            sentence = re.sub(
                r"^[A-Z][A-Z\s-]{8,}\s+"
                r"(?=[A-Z][a-z][A-Za-z\s-]{1,40}:)",
                "",
                sentence
            )
            words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", sentence)
            letters = re.findall(r"[A-Za-z]", sentence)
            normalized = sentence.casefold()
            if (
                len(words) < 8
                or len(words) > 55
                or len(sentence) > 360
                or not sentence[:1].isupper()
                or sentence[-1:] not in ".!?"
                or sentence.lower().startswith((
                    "unit ", "chapter ", "introduction to ",
                    "notes prepared", "prepared by"
                ))
                or (
                    len(letters) > 12
                    and sum(char.isupper() for char in letters) / len(letters) > 0.6
                )
                or sum(not char.isalnum() and not char.isspace() for char in sentence)
                > len(sentence) * 0.08
                or normalized in seen
            ):
                continue
            sentence_terms = study_topic_terms(sentence)
            overlap = len(topic_terms & sentence_terms)
            if topic_terms and not overlap:
                continue
            seen.add(normalized)
            points.append({
                "text": sentence,
                "topic": topic.strip() or "the study notes",
                "topic_overlap": overlap,
            })

    if topic_terms:
        points.sort(key=lambda point: point["topic_overlap"], reverse=True)
    return points


def generate_mcqs_from_notes(
    chunks,
    count,
    topic="",
    difficulty="Exam level",
    marks="Not applicable"
):
    """Create answerable MCQs from note statements and honor the requested count."""
    candidates = extract_study_points(chunks, topic)
    distractors = extract_study_points(chunks)
    if len(distractors) < 4:
        return []

    facts = extract_definition_facts(chunks)
    suggested_topics = suggest_topics(chunks)

    def label_for(candidate):
        if topic.strip():
            return topic.strip()
        sentence_terms = study_topic_terms(candidate)
        related_facts = [
            fact for fact in facts
            if study_topic_terms(fact["subject"]) & sentence_terms
            and len(fact["subject"].split()) <= 5
            and "," not in fact["subject"]
        ]
        if related_facts:
            return related_facts[0]["subject"]
        related_topics = [
            name for name in suggested_topics
            if study_topic_terms(name) & sentence_terms
        ]
        return related_topics[0] if related_topics else "the study notes"

    questions = []
    for candidate in candidates:
        answer = candidate["text"]
        question_topic = label_for(answer)
        question_terms = study_topic_terms(question_topic)
        distractor_pool = [
            item["text"] for item in distractors
            if item["text"] != answer
            and not (question_terms & study_topic_terms(item["text"]))
        ]
        if len(distractor_pool) < 3:
            distractor_pool = [
                item["text"] for item in distractors
                if item["text"] != answer
            ]
        if len(distractor_pool) < 3:
            continue

        offset = len(questions) * 3
        options = [answer] + [
            distractor_pool[(offset + index) % len(distractor_pool)]
            for index in range(3)
        ]
        if len(set(options)) != 4:
            continue
        correct_index = len(questions) % 4
        options.insert(correct_index, options.pop(0))
        if difficulty == "Easy":
            question = f"Which statement describes {question_topic} in the notes?"
        elif difficulty == "Challenging":
            question = (
                f"Which source-backed statement correctly characterizes "
                f"{question_topic}?"
            )
        else:
            question = (
                f"Which statement best explains {question_topic} according "
                "to the notes?"
            )
        if marks != "Not applicable":
            question = f"{question} ({marks})"
        questions.append({
            "question": question,
            "letters": list("ABCD"),
            "options": options,
            "answer_index": correct_index,
            "explanation": f"This statement is taken from the notes: {answer}"
        })
        if len(questions) >= count:
            break
    return questions


def infer_study_topic(text, requested_topic, topics, facts):
    """Choose a source-backed label for a generated question."""
    if requested_topic.strip():
        return requested_topic.strip()
    text_terms = study_topic_terms(text)
    for fact in facts:
        if study_topic_terms(fact["subject"]) & text_terms:
            subject = fact["subject"]
            if (
                len(subject.split()) <= 5
                and "," not in subject
                and ":" not in subject
            ):
                return subject
    for name in topics:
        if study_topic_terms(name) & text_terms:
            return name
    return "this point in the notes"


def build_local_exam_material(
    question_type,
    topic,
    count,
    chunks,
    marks="Not applicable",
    answer_style="Detailed model answer",
    difficulty="Exam level"
):
    """Build every exam-studio format directly from extracted PDF statements."""
    points = extract_study_points(chunks, topic)
    if not points:
        return "", [], [], "The PDF has no readable statements for this topic."

    selected = points[:count]
    facts = extract_definition_facts(chunks)
    topics = suggest_topics(chunks)
    labels = [
        infer_study_topic(point["text"], topic, topics, facts)
        for point in selected
    ]
    warning = ""
    if len(selected) < count:
        warning = (
            f"Found {len(selected)} distinct, readable statement(s) for "
            f"this topic; requested {count}."
        )

    if question_type == "Multiple-choice questions (MCQs)":
        questions = generate_mcqs_from_notes(
            chunks, count, topic, difficulty, marks
        )
        if len(questions) < count:
            warning = (
                f"Created {len(questions)} of {count} requested MCQs; "
                "the notes did not contain enough distinct statements and "
                "distractors for more reliable questions."
            )
        return (
            format_mcqs_for_display(questions),
            questions,
            [],
            warning
        )

    if question_type == "Point-wise revision notes":
        heading = topic.strip() or "Key revision points"
        material = f"## {heading}\n\n" + "\n".join(
            f"- {point['text']}" for point in selected
        )
        return material, [], [], warning

    if question_type == "Flashcards (Q&A cards)":
        sections = []
        for index, (point, label) in enumerate(zip(selected, labels), start=1):
            if difficulty == "Easy":
                prompt = f"What do the notes say about {label}?"
            elif difficulty == "Challenging":
                prompt = f"What key fact about {label} is supported by the notes?"
            else:
                prompt = f"Explain the point about {label} in the notes."
            sections.append(
                f"**Card {index}**\n"
                f"Q: {prompt}\n"
                f"A: {point['text']}"
            )
        return "\n\n".join(sections), [], parse_flashcards(
            "\n\n".join(sections)
        ), warning

    sections = []
    passage = "\n".join(point["text"] for point in selected)
    if question_type == "Passage-based questions":
        sections.extend(["### Passage 1", passage, ""])

    for index, (point, label) in enumerate(zip(selected, labels), start=1):
        if question_type == "Long-answer questions":
            question = f"Explain what the notes say about {label}."
        elif question_type == "Passage-based questions":
            question = f"What does Passage 1 state about {label}?"
        elif question_type == "Mixed questions":
            templates = (
                f"What do the notes state about {label}?",
                f"Explain the point about {label} in the notes.",
                f"Which key fact about {label} should you remember?",
            )
            question = templates[(index - 1) % len(templates)]
        else:
            if difficulty == "Easy":
                question = f"What do the notes state about {label}?"
            elif difficulty == "Challenging":
                question = (
                    f"Which key fact about {label} is supported by the notes?"
                )
            else:
                question = f"Explain the point about {label} in the notes."

        if marks != "Not applicable":
            question = f"{question} ({marks})"
        answer_points = [point["text"]]
        if (
            question_type == "Long-answer questions"
            and answer_style != "Brief answer"
        ):
            extra_point_count = {
                "2 marks": 1,
                "5 marks": 2,
                "10 marks": 4,
                "Mixed marks": 2,
            }.get(marks, 2)
            point_terms = study_topic_terms(point["text"])
            related_points = [
                candidate for candidate in points
                if candidate["text"] != point["text"]
                and len(
                    point_terms & study_topic_terms(candidate["text"])
                ) >= 2
            ]
            answer_points.extend(
                candidate["text"]
                for candidate in related_points[:extra_point_count]
            )
        answer = "\n".join(answer_points)
        if answer_style == "Point-wise answer":
            answer = "\n".join(
                f"- {answer_point}" for answer_point in answer_points
            )

        sections.extend([
            f"### Question {index}",
            question,
            f"**Model answer:** {answer}",
            ""
        ])

    return "\n".join(sections).strip(), [], [], warning


def format_mcqs_for_display(questions):
    """Format structured quiz questions with a separate answer key."""
    sections = []
    for index, question in enumerate(questions, start=1):
        lines = [
            f"### Question {index}",
            question["question"]
        ]
        lines.extend(
            f"{letter}. {option}"
            for letter, option in zip(
                question["letters"], question["options"]
            )
        )
        correct_letter = question["letters"][question["answer_index"]]
        lines.extend([
            f"**Correct answer:** {correct_letter}",
            f"**Explanation:** {question['explanation']}"
        ])
        sections.append("\n\n".join(lines))
    return "\n\n".join(sections)


def reset_quiz():
    """Clear quiz selections before the next Streamlit rerun renders radios."""
    for q_index in range(len(st.session_state.get("quiz_questions", []))):
        st.session_state.pop(f"quiz_choice_{q_index}", None)
    st.session_state["quiz_submitted"] = False


def parse_mcqs(text):
    """Parse generated MCQ text into structured questions for quiz mode.

    Returns a list of dicts (question, letters, options, answer_index,
    explanation). Returns [] when nothing parseable is found, so the caller
    can fall back to the plain Markdown view.
    """
    # Reuse the display normalizer: it rejoins wrapped bold labels and
    # splits options glued onto the question line.
    lines = format_for_display(text).splitlines()

    starts = [
        i for i, line in enumerate(lines)
        if re.match(
            r"^\s*(?:#{1,4}\s*)?(?:\*\*)?\s*question\s*\d+",
            line, re.IGNORECASE
        )
    ]
    questions = []
    for block_index, start in enumerate(starts):
        end = (
            starts[block_index + 1]
            if block_index + 1 < len(starts)
            else len(lines)
        )
        block = lines[start:end]

        remainder = re.sub(
            r"^\s*(?:#{1,4}\s*)?(?:\*\*)?\s*question\s*\d+\s*[\.\):\-:]?\s*(?:\*\*)?\s*",
            "", block[0], flags=re.IGNORECASE
        ).strip()
        question_text = remainder.strip("* ").strip()

        options = {}
        answer_letter = None
        explanation = ""

        for line in block[1:]:
            if not question_text:
                candidate = line.strip()
                is_option = re.match(
                    r"^\s*(?:[-*•]\s*)?(?:\*\*)?[A-Da-d](?:\*\*)?\s*[.):]",
                    candidate
                )
                is_answer = re.search(r"(?i)correct answer\s*:", candidate)
                is_explanation = re.search(r"(?i)explanation\s*:", candidate)
                if candidate and not is_option and not is_answer and not is_explanation:
                    question_text = candidate.strip("* ").strip()
                    continue

            answer_match = re.search(
                r"(?i)(?:\*\*)?\s*correct answer(?:\*\*)?\s*[:\-]\s*(.*)",
                line
            )
            explanation_match = re.search(
                r"(?i)(?:\*\*)?\s*explanation(?:\*\*)?\s*[:\-]\s*(.*)",
                line
            )
            option_match = re.match(
                r"^\s*(?:[-*•]\s*)?(?:\*\*)?([A-Da-d])(?:\*\*)?\s*[.):]\s+(.+)$",
                line
            )
            if answer_match:
                found = re.search(r"\b([A-Da-d])\b", answer_match.group(1))
                if found:
                    answer_letter = found.group(1).upper()
            elif explanation_match:
                if explanation_match.group(1).strip():
                    explanation = explanation_match.group(1).strip()
            elif option_match:
                letter = option_match.group(1).upper()
                if letter not in options and len(options) < 4:
                    options[letter] = option_match.group(2).strip()

        letters_ordered = [letter for letter in "ABCD" if letter in options]
        if (
            not question_text
            or len(letters_ordered) < 2
            or answer_letter not in letters_ordered
        ):
            continue

        questions.append({
            "question": question_text,
            "letters": letters_ordered,
            "options": [options[letter] for letter in letters_ordered],
            "answer_index": letters_ordered.index(answer_letter),
            "explanation": explanation
        })
    return questions


def parse_flashcards(text):
    """Parse 'Card N' / 'Q: ...' / 'A: ...' output into question-answer pairs.

    Returns a list of dicts (question, answer). Unparseable output yields
    [], and the caller falls back to plain text display.
    """
    pairs = []
    current = None

    def finish():
        if current is not None and current["q"] and current["a_parts"]:
            return {
                "question": current["q"],
                "answer": " ".join(
                    part for part in current["a_parts"] if part
                )
            }
        return None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        q_match = re.match(
            r"^(?:\*\*)?Q(?:uestion)?\s*\d*\s*(?:\*\*)?\s*[:.]\s*(.+)$",
            line, re.IGNORECASE
        )
        a_match = re.match(
            r"^(?:\*\*)?A(?:nswer)?\s*\d*\s*(?:\*\*)?\s*[:.]\s*(.+)$",
            line, re.IGNORECASE
        )
        card_match = re.match(
            r"^(?:\*\*)?card\s*\d+(?:\*\*)?[:.]?\s*$",
            line, re.IGNORECASE
        )
        if q_match:
            completed = finish()
            if completed:
                pairs.append(completed)
            current = {"q": q_match.group(1).strip(), "a_parts": []}
        elif a_match and current is not None:
            current["a_parts"].append(a_match.group(1).strip())
        elif card_match and current is not None:
            completed = finish()
            if completed:
                pairs.append(completed)
            current = {"q": "", "a_parts": []}
        elif current is not None and current["a_parts"]:
            current["a_parts"].append(line)

    completed = finish()
    if completed:
        pairs.append(completed)
    return pairs


def generate_flashcards(topic, count, difficulty, context):
    """Generate Q/A flashcards from the supplied passages."""
    topic_text = (topic or "").strip()
    if topic_text:
        topic_line = (
            f'Every card must be about ONLY the topic "{topic_text}" as '
            "explained in the passages. If the passages say little about "
            "it, make fewer cards about what is actually present. Never "
            "fill gaps with other topics."
        )
    else:
        topic_line = (
            "Cards may cover the main topics present in the passages."
        )

    prompt = f"""
You are StudyMate AI, creating exam flashcards for a student.

Create exactly {count} flashcards. Difficulty: {difficulty}.

{topic_line}

FORMAT (follow exactly):
**Card 1**
Q: [one clear question]
A: [concise answer, 1-3 sentences]

**Card 2**
Q: [one clear question]
A: [concise answer, 1-3 sentences]

RULES:
- Put Q and A each on their own line, exactly as shown.
- Answers must come only from the supplied passages.
- Use simple English; briefly explain technical terms inside the answer.
- Do not invent facts or examples.

SOURCE PASSAGES:
{context}
"""
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You create faithful flashcards from the supplied "
                    "notes. Follow the requested card format exactly."
                )
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0, "num_ctx": 8192}
    )
    return response["message"]["content"].strip()


# ==========================================
# 8. TABS FOR CHAT, SUMMARY, AND EXAM PREPARATION
# ==========================================
chat_tab, summary_tab, exam_tab = st.tabs([
    "💬 Ask Your Notes",
    "📖 PDF Summary",
    "🎓 Exam Preparation Studio"
])

# ==========================================
# 9. CHAT FEATURE
# ==========================================
with chat_tab:
    st.subheader("Ask questions about your notes")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(format_for_display(message["content"]))

    question = st.chat_input("Ask a question about your notes...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your notes..."):
                    previous_messages = st.session_state.messages[:-1]
                    recent_history = previous_messages[-4:]

                    search_text = question
                    if len(question.split()) < 6 and recent_history:
                        last_user_question = next(
                            (m["content"] for m in reversed(recent_history)
                             if m["role"] == "user"),
                            ""
                        )
                        if last_user_question:
                            search_text = f"{last_user_question} {question}"

                    passages, distances, _ = retrieve_context(
                        search_text,
                        number_of_chunks=5
                    )

                    if not passages:
                        answer = (
                            "I couldn't find relevant information "
                            "in this document."
                        )
                    else:
                        context = make_context(passages)
                        answer = generate_chat_answer(
                            question,
                            context,
                            history=recent_history
                        )

                st.markdown(format_for_display(answer))

                with st.expander("View retrieved passages"):
                    for i, passage in enumerate(passages):
                        distance = distances[i] if i < len(distances) else None
                        distance_label = (
                            f" — Distance: {distance:.4f}"
                            if distance is not None else ""
                        )
                        st.markdown(f"**Passage {i + 1}{distance_label}**")
                        st.write(passage)

            except Exception as error:
                answer = (
                    "Could not answer this question. Check that Ollama is "
                    f"running and both models are available.\n\nDetails: {error}"
                )
                st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# ==========================================
# 10. PDF SUMMARY STUDIO
# ==========================================
with summary_tab:
    st.subheader("📖 Understand your PDF faster")
    st.write(
        "Generate a clear summary of the complete uploaded document, "
        "including its main ideas, definitions, and revision points."
    )

    summary_exists = (
        st.session_state.get("summary_document") == file_hash
        and st.session_state.get("summary_version") == SUMMARY_VERSION
        and bool(st.session_state.get("document_summary"))
    )

    if not summary_exists:
        st.info("Your PDF is ready. Generate a summary whenever you are ready to study.")

    if st.button(
        "✨ Generate complete PDF summary"
        if not summary_exists else "🔄 Regenerate summary",
        type="primary",
        use_container_width=True,
        key="generate_pdf_summary"
    ):
        try:
            with st.spinner("Extracting key points from your PDF..."):
                summary = generate_document_summary()
                st.session_state.document_summary = summary
                st.session_state.summary_document = file_hash
                st.session_state.summary_version = SUMMARY_VERSION
            st.success("Your quick PDF summary is ready!")
        except Exception as error:
            st.error(f"Could not create the PDF summary: {error}")

    if (
        st.session_state.get("summary_document") == file_hash
        and st.session_state.get("summary_version") == SUMMARY_VERSION
        and st.session_state.get("document_summary")
    ):
        st.divider()
        st.markdown(st.session_state.document_summary)
        st.download_button(
            "⬇️ Download PDF summary",
            data=st.session_state.document_summary,
            file_name="learnnova_pdf_summary.md",
            mime="text/markdown",
            key="download_pdf_summary"
        )


# ==========================================
# 11. EXAM PREPARATION PROMPT BUILDER AND GENERATOR
# ==========================================
def build_exam_prompt(
    question_type,
    topic,
    marks,
    answer_style,
    difficulty,
    count,
    context
):
    """Create strict instructions tailored to the selected output format."""

    topic = (topic or "").strip()

    if question_type == "Multiple-choice questions (MCQs)":
        format_instructions = """
FORMAT: MULTIPLE-CHOICE QUESTIONS ONLY.
For every item, use this exact structure:
### Question 1
**[Question text]**
A. [Option]
B. [Option]
C. [Option]
D. [Option]
**Correct answer:** [letter and full option text]
**Explanation:** [why the answer is supported by the notes]

Rules:
- Every question must have exactly four options labelled A, B, C, D.
- There must be exactly one best correct answer per question.
- Make the three incorrect options plausible, but contradicted or unsupported
  by the source material.
- Include the correct answer and explanation after every question.
- Do not produce a summary, revision notes, or essay instead of MCQs.
- Do not label the output as multiple-select questions.
IMPORTANT FORMATTING RULES:
- Put each question on a separate line.
- Put options A, B, C, and D on four separate lines.
- Put the correct answer on its own line.
- Put the explanation on its own line.
- Never combine multiple options on the same line.
- Leave a blank line between questions.
"""
    elif question_type == "Passage-based questions":
        format_instructions = """
FORMAT: PASSAGE-BASED QUESTIONS WITH ANSWERS.
Create a short source-based passage for each set using only information
faithfully paraphrased from the supplied notes. Do not put invented quotations
in quotation marks or pretend a paraphrase is an exact quotation.
After each passage, create questions that require students to use that passage.
Provide a model answer for every question. Label the passage, questions,
answers, and marks clearly. Do not produce MCQs unless they are explicitly
part of the requested passage-based format.
"""
    elif question_type == "Point-wise revision notes":
        format_instructions = """
FORMAT: TOPIC-FOCUSED EXAM REVISION NOTES.

The student has selected ONE topic. These notes must be about that topic
only, not about the whole subject.

Requirements:
1. Begin with the exact selected topic as the main heading.
2. Give a clear, accurate definition of the topic using the supplied notes.
3. Organize the content under relevant numbered subheadings, such as:
   - Meaning and definition
   - Key characteristics
   - Types or categories within the topic
   - How it works (steps or process)
   - Examples explicitly supported by the notes
   - Advantages and limitations
   - Applications
   - Difference from closely related concepts
   Include a subheading only if the supplied passages support it for this
   topic. Skip the rest.
4. Under each subheading, write short numbered points in simple English.
   Briefly explain every technical term the first time it appears.
5. Every point must be about the selected topic and supported by the
   supplied passages. Mention other concepts only when a passage uses them
   to explain this topic.
6. Explain differences clearly when the notes compare this topic with a
   related one (for example, two categories of the same concept).
7. End with a section called "Likely exam questions": 3 to 5 short exam
   questions on this topic with concise answers, based only on the notes.
8. Do not begin with phrases like "Here is a summary".
9. Do not write a general overview of the whole subject.
10. Do not invent facts, examples, quotations, or page references. If the
    document lacks information on part of the topic, add a short section
    "Not covered in this document" listing exactly what is missing.

Return revision notes only, not a multiple-choice paper.
"""
    elif question_type == "Short-answer questions":
        format_instructions = """
FORMAT: SHORT-ANSWER QUESTIONS AND MODEL ANSWERS.
Number every question. Put the model answer immediately after it.
Keep answers concise and suitable for the selected marks and answer style.
Use a definition plus the key points needed to answer the question.
Do not produce MCQ options or a general summary.
"""
    elif question_type == "Long-answer questions":
        format_instructions = """
FORMAT: LONG-ANSWER EXAM QUESTIONS AND MODEL ANSWERS.
Number every question. Give each a structured model answer with a suitable
introduction/definition, headings, explained key points, and a conclusion
when appropriate. Use examples only when supported by the notes.
Make answers detailed enough for the selected marks and answer style.
Do not produce MCQs or a general revision summary.
"""
    else:
        format_instructions = """
FORMAT: A MIXED QUESTION-AND-ANSWER SET.
Create a varied combination of suitable exam question types, such as short
answers, explanations, comparisons, applications, and MCQs where appropriate.
Clearly label the type of each question and include a model answer for every
item. Do not make every item the same type. Do not return revision notes alone.
"""

    if topic:
        topic_instruction = (
            f'The student selected ONE topic: "{topic}".\n'
            "Focus ONLY on this topic. Every heading, point, question, and "
            f"answer must be about \"{topic}\" as explained in the supplied "
            "passages.\n"
            "Do not cover other topics from the passages, even briefly.\n"
            "If the passages contain little or nothing about this topic, "
            "cover only what the passages actually say about it and clearly "
            "state which parts the document does not provide. Never fill the "
            "gap with other topics."
        )
    else:
        topic_instruction = (
            "No specific topic was selected. Cover the main topics present "
            "in the supplied source passages."
        )

    if question_type == "Point-wise revision notes":
        task_instruction = (
            "Generate topic-focused, exam-ready revision notes in the "
            "selected format. The number of items does not apply to revision "
            "notes; do not pad the notes with unrelated material to reach a "
            "count."
        )
    else:
        task_instruction = (
            f"Generate exactly {count} items in the selected format: "
            f"{question_type}. The selected format is mandatory. Do not "
            "substitute another format."
        )

    if marks == "Not applicable":
        mark_instruction = "Marks are not applicable to revision notes."
    elif marks == "Mixed marks":
        mark_instruction = (
            "Use a sensible mix of 2-, 5-, and 10-mark questions. "
            "Show the mark allocation beside every question."
        )
    else:
        mark_instruction = (
            f"Each question is worth {marks}. Show the mark allocation. "
            "Match the depth and length of the answer to these marks."
        )

    if topic:
        fallback_rule = (
            "- If the supplied passages do not support part of the selected "
            "topic, state clearly that the document does not provide that "
            "information. Never replace it with other topics."
        )
    else:
        fallback_rule = (
            "- If the passages do not support a potential question and "
            "answer, choose a different supported topic instead."
        )

    if question_type == "Point-wise revision notes":
        finish_rule = (
            "Write complete notes; do not stop after an introduction or a "
            "definition."
        )
    else:
        finish_rule = (
            f"Finish all {count} items; do not stop after an introduction."
        )

    if topic:
        topic_reminder = (
            f'BEFORE WRITING, RE-READ THE TOPIC: "{topic}". Use only passage '
            f"material about \"{topic}\". Skip passage material about other "
            "topics. If the passages say little about this topic, say what "
            "is missing instead of writing about other topics."
        )
    else:
        topic_reminder = (
            "No topic was selected, so cover the main topics in the passages."
        )

    return f"""
You are StudyMate AI, an academic exam-preparation assistant.

TASK:
{task_instruction}

TOPIC:
{topic_instruction}

DIFFICULTY:
{difficulty}. Easy means basic recall and understanding; Exam level means
typical academic exam questions; Challenging means deeper comparison or
reasoning, but only where the source supports it.

ANSWER STYLE:
{answer_style}. Apply this style to every model answer where answers are
requested. If the student selected "Examples where supported by notes",
include examples only when the source actually supports them.

MARKS:
{mark_instruction}

REQUIRED OUTPUT FORMAT:
{format_instructions}

SOURCE AND ACCURACY RULES:
- Use only the supplied passages for claims about the uploaded notes.
- Do not invent facts, examples, quotations, page numbers, or source references.
- Paraphrase faithfully; do not change the meaning of the source.
{fallback_rule}
- Avoid duplicate questions and duplicate answers.
- Clearly separate every item so the student can study from it.
- For question formats, every question must have its model answer.
- Do not claim that questions are guaranteed to appear in a real examination.
- Do not begin with a generic summary. Start directly with the selected format.
- {finish_rule}

TOPIC FOCUS REMINDER:
{topic_reminder}

SOURCE PASSAGES:
{context}
"""


def output_matches_format(output, question_type, count):
    """Check basic format markers for every supported exam output type."""
    text = output.strip()
    if not text:
        return False, "The model returned an empty response."

    question_headings = len(re.findall(
        r"(?im)^\s*(?:#{1,4}\s*)?(?:question\s+)?\d+"
        r"(?:\s*[\.\):\-]|\s*$)",
        text
    ))
    answer_labels = len(re.findall(
        r"(?i)(?:\*\*)?(?:model answer|correct answer|answer key|answer)"
        r"(?:\*\*)?\s*:",
        text
    ))

    if question_type == "Multiple-choice questions (MCQs)":
        # Accept A. text, A) text, A: text, including bullet-prefixed options.
        option_count = len(re.findall(
            r"(?im)^\s*(?:[-*•]\s*)?(?:\*\*)?(?:A|B|C|D)(?:\*\*)?\s*[\.\):]\s+",
            text
        ))
        explanation_count = len(re.findall(
            r"(?im)^\s*(?:[-*•]\s*)?(?:\*\*)?explanation(?:\*\*)?\s*:",
            text
        ))
        if option_count < count * 4 or answer_labels < count:
            return False, (
                f"MCQ check failed: found {option_count} option lines and "
                f"{answer_labels} answer labels for {count} questions."
            )
        if explanation_count < count:
            return False, "Each MCQ must also include an explanation."

    elif question_type == "Short-answer questions":
        if question_headings < count or answer_labels < count:
            return False, (
                f"Short-answer check failed: expected {count} numbered "
                "questions and a labelled model answer for each."
            )

    elif question_type == "Long-answer questions":
        if question_headings < count or answer_labels < count:
            return False, (
                f"Long-answer check failed: expected {count} numbered "
                "questions and a labelled model answer for each."
            )
        if len(re.findall(r"(?im)^\s*#{1,4}\s+", text)) < 2:
            return False, "Long answers should use headings to structure explanations."

    elif question_type == "Passage-based questions":
        passage_count = len(re.findall(r"(?i)\bpassage\s*\d*\s*:?", text))
        if passage_count < 1 or question_headings < count or answer_labels < count:
            return False, (
                f"Passage-based check failed: expected a labelled passage, "
                f"{count} numbered questions, and model answers."
            )

    elif question_type == "Point-wise revision notes":
        bullet_count = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[\.\)])\s+", text))
        heading_count = len(re.findall(r"(?m)^\s*#{1,4}\s+", text))
        if bullet_count < 3 or heading_count < 1:
            return False, (
                "Revision-notes check failed: expected topic headings and "
                "at least three bullet points. Do not return a prose summary."
            )

    elif question_type == "Mixed questions":
        if question_headings < count or answer_labels < count:
            return False, (
                f"Mixed-question check failed: expected {count} numbered "
                "questions and a labelled model answer for each."
            )
        # Mixed output may include MCQs, but must not be only a summary.
        if not re.search(r"(?i)\b(question|model answer|correct answer)\b", text):
            return False, "The output does not look like a question-and-answer set."

    return True, ""


def generate_exam_material(
    question_type,
    topic,
    marks,
    answer_style,
    difficulty,
    count,
    context,
    chunks=None
):
    """Generate requested study material locally from PDF-backed statements."""
    source_chunks = chunks if chunks is not None else [context]
    output, questions, cards, warning = build_local_exam_material(
        question_type,
        topic,
        count,
        source_chunks,
        marks,
        answer_style,
        difficulty
    )
    actual_count = len(questions) or len(cards)
    if question_type == "Point-wise revision notes":
        actual_count = len(re.findall(r"(?m)^\s*-\s+", output))
    elif not actual_count:
        actual_count = len(re.findall(
            r"(?im)^\s*#{1,4}\s*question\s+\d+\b",
            output
        ))
    if actual_count < count and not warning:
        warning = (
            f"Created {actual_count} of {count} requested item(s); "
            "the notes did not contain enough distinct source statements."
        )

    valid, reason = output_matches_format(
        output,
        question_type,
        actual_count or count
    )
    if not output:
        return output, False, warning or reason
    if not valid:
        warning = f"{warning} {reason}".strip()
    return output, valid, warning


# ==========================================
# 11. EXAM PREPARATION STUDIO
# ==========================================
with exam_tab:
    st.subheader("🎓 Exam-Ready Questions & Answers")
    st.write(
        "Choose the study material you need. StudyMate AI will tailor its "
        "output to the selected format, topic, marks, and difficulty."
    )
    st.caption(
        "Every format is generated from your PDF notes, with the requested "
        "item count checked. Answers and examples stay limited to what the "
        "notes support."
    )

    # ---------- Topic picker: discover and quick-select topics ----------
    if (
        st.session_state.get("topics_doc") != file_hash
        or st.session_state.get("topics_version") != TOPIC_SUGGESTION_VERSION
    ):
        st.session_state.pop("topic_suggestions", None)
        st.session_state.topics_doc = file_hash
        st.session_state.topics_version = TOPIC_SUGGESTION_VERSION

    if "topic_suggestions" not in st.session_state:
        st.session_state.topic_suggestions = suggest_topics(
            get_all_document_chunks()
        )

    if st.session_state.get("topic_suggestions"):
        st.write("Topics found in your notes — select one or type your own:")
        suggestions = st.session_state.topic_suggestions
        for row_start in range(0, len(suggestions), 4):
            row = suggestions[row_start:row_start + 4]
            cols = st.columns(4)
            for col_index, (col, name) in enumerate(zip(cols, row)):
                if col.button(name, key=f"topic_btn_{row_start}_{col_index}"):
                    st.session_state["exam_topic"] = name

    with st.form("exam_generator_form"):
        topic = st.text_input(
            "Topic (optional)",
            placeholder="e.g., supervised learning; leave blank for broad coverage",
            key="exam_topic"
        )

        question_type = st.selectbox(
            "What would you like to generate?",
            [
                "Mixed questions",
                "Multiple-choice questions (MCQs)",
                "Passage-based questions",
                "Point-wise revision notes",
                "Short-answer questions",
                "Long-answer questions",
                "Flashcards (Q&A cards)"
            ],
            index=1
        )

        mark_options = ["2 marks", "5 marks", "10 marks", "Mixed marks"]
        if question_type in ("Point-wise revision notes", "Flashcards (Q&A cards)"):
            marks = "Not applicable"
            st.info("This format does not require marks.")
        else:
            marks = st.selectbox("Marks per question", mark_options, index=1)

        answer_style = st.selectbox(
            "Answer style",
            [
                "Detailed model answer",
                "Point-wise answer",
                "Brief answer",
                "Examples where supported by notes"
            ]
        )

        difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Exam level", "Challenging"],
            index=1
        )

        count = st.select_slider(
            "Number of questions or items",
            options=[5, 10, 15, 20, 25, 30],
            value=5
        )

        generate_button = st.form_submit_button(
            "✨ Generate Study Material",
            type="primary",
            use_container_width=True
        )

    if generate_button:
        for state_key in [
            "exam_material",
            "exam_document",
            "exam_sources",
            "exam_distances",
            "exam_format",
            "exam_format_valid",
            "exam_format_warning",
            "quiz_questions",
            "quiz_submitted",
            "flashcard_pairs"
        ]:
            st.session_state.pop(state_key, None)

        try:
            with st.spinner("Preparing material from your PDF notes..."):
                document_chunks = get_all_document_chunks()
                exam_material, format_valid, format_warning = (
                    generate_exam_material(
                        question_type=question_type,
                        topic=topic,
                        marks=marks,
                        answer_style=answer_style,
                        difficulty=difficulty,
                        count=count,
                        context="",
                        chunks=document_chunks
                    )
                )

            if question_type == "Multiple-choice questions (MCQs)":
                quiz_questions = parse_mcqs(exam_material)
                st.session_state["quiz_questions"] = quiz_questions
                st.session_state["quiz_submitted"] = False
                for q_index in range(len(quiz_questions)):
                    st.session_state.pop(f"quiz_choice_{q_index}", None)
                if not quiz_questions:
                    format_valid = False
                    format_warning = (
                        format_warning
                        or "The generated multiple-choice questions could not "
                        "be parsed for the interactive quiz."
                    )
            elif question_type == "Flashcards (Q&A cards)":
                cards = parse_flashcards(exam_material)
                st.session_state["flashcard_pairs"] = cards
                if len(cards) != min(count, len(extract_study_points(
                    document_chunks, topic
                ))):
                    format_valid = False
                    format_warning = (
                        format_warning
                        or "The generated flashcards could not all be parsed."
                    )
            else:
                st.session_state.pop("quiz_questions", None)
                st.session_state.pop("quiz_submitted", None)
                st.session_state.pop("flashcard_pairs", None)

            source_points = extract_study_points(document_chunks, topic)
            if not source_points:
                source_points = extract_study_points(document_chunks)[:5]
            st.session_state["exam_material"] = exam_material
            st.session_state["exam_document"] = file_hash
            st.session_state["exam_sources"] = [
                point["text"] for point in source_points[:5]
            ]
            st.session_state["exam_distances"] = []
            st.session_state["exam_format"] = question_type
            st.session_state["exam_format_valid"] = format_valid
            st.session_state["exam_format_warning"] = format_warning
            if not exam_material:
                st.warning(
                    format_warning
                    or "No readable source statements were available for this request."
                )
            elif format_warning and format_valid:
                st.info(format_warning)

        except Exception as error:
            st.error(
                f"Could not generate study material: {error}"
            )

    # Show saved results in this tab; they survive reruns via session state.
    if st.session_state.get("exam_document") == file_hash:
        exam_material = st.session_state.get("exam_material")

        if exam_material:
            st.divider()
            st.subheader("📝 Your Study Material")

            if not st.session_state.get("exam_format_valid", True):
                st.error(
                    "The generated response did not pass the format check "
                    "for your selected option. Please review it carefully; "
                    "it may be incomplete or in the wrong format. Try again "
                    "with a smaller item count or a more specific topic. "
                    "Details: "
                    + st.session_state.get(
                        "exam_format_warning",
                        "The output structure could not be verified."
                    )
                )

            is_mcq_quiz = (
                st.session_state.get("exam_format")
                == "Multiple-choice questions (MCQs)"
                and bool(st.session_state.get("quiz_questions"))
            )
            is_flashcards = (
                st.session_state.get("exam_format") == "Flashcards (Q&A cards)"
                and bool(st.session_state.get("flashcard_pairs"))
            )

            if is_mcq_quiz:
                st.markdown(
                    "Choose an answer for each question, then submit to see "
                    "your score and explanations."
                )
                quiz_questions = st.session_state["quiz_questions"]
                for q_index, item in enumerate(quiz_questions):
                    choices = [
                        f"{letter}. {text}"
                        for letter, text in zip(
                            item["letters"], item["options"]
                        )
                    ]
                    st.radio(
                        f"Question {q_index + 1}: {item['question']}",
                        options=choices,
                        key=f"quiz_choice_{q_index}",
                        index=None
                    )

                col_submit, col_reset = st.columns(2)
                if col_submit.button("✅ Submit answers", key="submit_quiz"):
                    st.session_state.quiz_submitted = True
                col_reset.button(
                    "🔄 Reset quiz",
                    key="reset_quiz",
                    on_click=reset_quiz
                )

                if st.session_state.get("quiz_submitted"):
                    score = 0
                    st.divider()
                    for q_index, item in enumerate(quiz_questions):
                        chosen = st.session_state.get(
                            f"quiz_choice_{q_index}"
                        )
                        correct_label = (
                            f"{item['letters'][item['answer_index']]}. "
                            + item["options"][item["answer_index"]]
                        )
                        if chosen == correct_label:
                            score += 1
                            st.success(f"Question {q_index + 1}: Correct!")
                        elif chosen is None:
                            st.info(
                                f"Question {q_index + 1}: Not answered. "
                                f"Correct answer: {correct_label}"
                            )
                        else:
                            st.error(
                                f"Question {q_index + 1}: Incorrect. "
                                f"Correct answer: {correct_label}"
                            )
                        st.caption(f"Explanation: {item['explanation']}")
                    st.markdown(
                        f"### 🏁 Score: {score} / {len(quiz_questions)}"
                    )

            if is_flashcards:
                for card_index, card in enumerate(
                    st.session_state["flashcard_pairs"], start=1
                ):
                    with st.expander(f"🎴 Card {card_index}: {card['question']}"):
                        st.markdown(card["answer"])
                with st.expander("View flashcards as plain text"):
                    st.markdown(format_for_display(exam_material))
            elif is_mcq_quiz:
                with st.expander("View answer key and explanations"):
                    st.markdown(format_for_display(exam_material))
            else:
                st.markdown(format_for_display(exam_material))

            st.download_button(
                "⬇️ Download Study Material",
                data=exam_material,
                file_name="studymate_exam_material.md",
                mime="text/markdown",
                key="download_exam_material"
            )

            with st.expander("View source passages used"):
                source_passages = st.session_state.get("exam_sources", [])
                source_distances = st.session_state.get("exam_distances", [])

                for i, passage in enumerate(source_passages):
                    distance_text = ""
                    if i < len(source_distances):
                        distance_text = (
                            f" — Distance: {source_distances[i]:.4f}"
                        )
                    st.markdown(f"**Source passage {i + 1}{distance_text}**")
                    st.write(passage)

st.divider()
st.caption(
    "StudyMate AI | Local AI with Ollama and ChromaDB. "
    "Always review generated answers against your original notes."
)