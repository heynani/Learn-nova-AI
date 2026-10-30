import streamlit as st
import ollama
import chromadb
import pymupdf
import hashlib
import re
import io

# Optional OCR support for scanned PDFs.
# If these imports fail, the app still works normally for text PDFs.
try:

    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    from PIL import Image
    OCR_AVAILABLE = True
    
    # If Tesseract is installed but not on PATH, uncomment and adjust:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    OCR_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Learnnova AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Learnnova AI")
st.caption("Your intelligent learning companion - learn, revise, and prepare smarter.")

# ==========================================
# 2. SETTINGS, DATABASE CONNECTION, EMBEDDING HELPERS
# ==========================================
CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text expects task prefixes ("search_document:", "search_query:").
# Bumping EMBEDDING_VERSION changes collection names, so vectors stored by
# older versions of this app (without prefixes) are never mixed with new ones.
EMBEDDING_VERSION = "v2"

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
            document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            full_text = "\n".join(page.get_text() for page in document).strip()

            if not full_text and OCR_AVAILABLE:
                st.caption(
                    "Scanned PDF detected - running OCR. This can take a "
                    "while for long documents..."
                )
                ocr_pages = []
                for page in document:
                    pix = page.get_pixmap(dpi=200)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_pages.append(pytesseract.image_to_string(image))
                full_text = "\n".join(ocr_pages).strip()

            document.close()
        except Exception as error:
            extraction_error = str(error)

    if extraction_error:
        st.error(f"Could not process this PDF: {extraction_error}")
        st.stop()

    if not full_text:
        if OCR_AVAILABLE:
            st.error(
                "No text could be read from this PDF, even with OCR. "
                "Please check that the file is a readable document."
            )
        else:
            st.error(
                "No selectable text was found in this PDF. To read scanned "
                "PDFs, install Tesseract OCR and pytesseract (see setup "
                "notes), then re-upload the file."
            )
        st.stop()

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
            st.error(f"Could not process this PDF: {error}")
            st.stop()

    st.success(f"Prepared {len(chunks)} chunks.")

st.caption(f"📄 Document: {st.session_state.library[file_hash]}")
st.caption(f"📚 Stored chunks: {collection.count()}")

if collection.count() == 0:
    st.warning("No text chunks are available in this document.")
    st.stop()


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
        "quiz_submitted"
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
    data = collection.get(include=["documents"])
    documents = data.get("documents") or []
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
    """One LLM call: list the main topics covered by the sampled excerpts."""
    sample = "\n\n".join(
        f"Excerpt {i + 1}:\n{passage[:600]}"
        for i, passage in enumerate(passages)
    )
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract concise study-topic names from notes."
            },
            {
                "role": "user",
                "content": (
                    "The following are excerpts from a student's study "
                    "notes. List the main study topics they cover.\n"
                    "Return ONLY one comma-separated list of short topic "
                    "names (2-5 words each). No numbering, no extra text.\n\n"
                    f"{sample}"
                )
            }
        ],
        options={"temperature": 0, "num_ctx": 4096}
    )
    raw = response["message"]["content"].strip()
    topics = []
    for part in re.split(r"[,\n]", raw):
        name = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", part.strip(" -•*\t"))
        name = name.strip()
        if 2 <= len(name) <= 40:
            if name.lower() not in {t.lower() for t in topics}:
                topics.append(name)
    return topics[:limit]


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
# 8. TABS FOR CHAT AND EXAM PREPARATION
# ==========================================
chat_tab, exam_tab = st.tabs([
    "💬 Ask Your Notes",
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
# 10. EXAM PREPARATION PROMPT BUILDER AND GENERATOR
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
        r"(?im)^\s*(?:#{1,4}\s*)?(?:question\s*)?\d+\s*[\.\):\-]",
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
    context
):
    """Generate once, validate the selected format, then retry if needed."""
    prompt = build_exam_prompt(
        question_type, topic, marks, answer_style,
        difficulty, count, context
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Follow the selected study-material format exactly. "
                    "Do not substitute a generic summary for a question set. "
                    "For every requested question, include its answer. "
                    "For revision notes, produce headings and bullet points "
                    "about the selected topic only. "
                    "Use only the supplied notes for source-specific facts."
                )
            },
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0, "num_ctx": 8192}
    )
    output = response["message"]["content"].strip()
    valid, reason = output_matches_format(output, question_type, count)

    if not valid:
        if question_type == "Multiple-choice questions (MCQs)":
            correction = f"""
Regenerate from scratch as exactly {count} MCQs. Each item must have a
numbered heading, A., B., C., D. options, **Correct answer:** and
**Explanation:**. No summary or revision notes.
"""
        elif question_type == "Passage-based questions":
            correction = f"""
Regenerate from scratch. Include at least one clearly labelled source-based
passage, then exactly {count} numbered questions, each followed by
**Model answer:**. Base the passage and answers only on the supplied notes.
Do not return a general summary.
"""
        elif question_type == "Point-wise revision notes":
            selected_topic = (topic or "").strip()
            if selected_topic:
                correction = f"""
Regenerate from scratch as revision notes, NOT questions, about ONLY the
topic "{selected_topic}". Start with that topic as the main heading, then use
Markdown subheadings and short bullet points from the supplied passages.
Do not cover other topics. If the passages contain little about this topic,
write what is available and add a section listing what the document does not
provide.
"""
            else:
                correction = """
Regenerate from scratch as revision notes, NOT questions. Include clear
Markdown topic headings and concise bullet points from the supplied notes.
Do not return an opening prose summary instead of notes.
"""
        elif question_type == "Short-answer questions":
            correction = f"""
Regenerate from scratch as exactly {count} numbered short-answer exam
questions. Immediately after every question, include **Model answer:** with
a concise answer appropriate to {marks} and {answer_style}. No summary.
"""
        elif question_type == "Long-answer questions":
            correction = f"""
Regenerate from scratch as exactly {count} numbered long-answer exam
questions. Each must have **Model answer:** followed by a detailed answer
with meaningful Markdown headings and explained key points. Match {marks},
{difficulty}, and {answer_style}. No summary.
"""
        else:
            correction = f"""
Regenerate from scratch as exactly {count} numbered mixed exam questions.
Use varied question types rather than repeating one type. Each question must
have a labelled **Model answer:** (or **Correct answer:** for MCQs).
Do not return revision notes or a generic summary.
"""

        retry_prompt = (
            prompt
            + "\n\nFORMAT VALIDATION FAILED:\n"
            + reason
            + "\n\nMANDATORY CORRECTION:\n"
            + correction
            + "\n\nReturn only the requested material and include every item."
        )
        retry = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Obey the selected format and correction requirements. "
                        "Do not replace the requested output with a summary."
                    )
                },
                {"role": "user", "content": retry_prompt}
            ],
            options={"temperature": 0, "num_ctx": 8192}
        )
        output = retry["message"]["content"].strip()
        valid, reason = output_matches_format(output, question_type, count)

    return output, valid, reason


# ==========================================
# 11. EXAM PREPARATION STUDIO
# ==========================================
with exam_tab:
    st.subheader("🎓 Exam-Ready Questions & Answers")
    st.write(
        "Choose the study material you need. StudyMate AI will tailor its "
        "output to the selected format, topic, marks, and difficulty."
    )

    # ---------- Topic picker: discover and quick-select topics ----------
    if st.session_state.get("topics_doc") != file_hash:
        st.session_state.pop("topic_suggestions", None)
        st.session_state.topics_doc = file_hash

    if not st.session_state.get("topic_suggestions"):
        if st.button("🔎 Find topics in this document", key="find_topics"):
            try:
                with st.spinner("Finding the main topics..."):
                    sample = get_topic_sample()
                    if sample:
                        st.session_state.topic_suggestions = suggest_topics(
                            sample
                        )
                    else:
                        st.warning("No stored text to analyze.")
            except Exception as error:
                st.warning(f"Could not suggest topics: {error}")

    if st.session_state.get("topic_suggestions"):
        st.write("Quick-select a topic:")
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
            ]
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
            value=10
        )

        generate_button = st.form_submit_button(
            "✨ Generate Study Material",
            type="primary",
            use_container_width=True
        )

    if generate_button:
        if topic.strip():
            search_text = topic.strip()
        else:
            search_text = (
                "Important definitions, concepts, processes, comparisons, "
                "advantages, limitations, applications, and key topics "
                "from the study notes."
            )

        try:
            with st.spinner("Retrieving notes and generating your material..."):
                chunk_count = 8 if topic.strip() else 12
                passages, distances, metadatas = retrieve_context(
                    search_text,
                    number_of_chunks=chunk_count
                )

                if not passages:
                    st.session_state.pop("exam_material", None)
                    st.warning(
                        "No relevant notes were retrieved for this request."
                    )
                else:
                    coverage = topic_coverage_message(topic, passages)
                    if coverage:
                        st.warning(coverage)

                    context = make_context(passages)

                    if question_type == "Flashcards (Q&A cards)":
                        raw_cards = generate_flashcards(
                            topic=topic,
                            count=count,
                            difficulty=difficulty,
                            context=context
                        )
                        parsed_cards = parse_flashcards(raw_cards)
                        exam_material = raw_cards
                        format_valid = bool(parsed_cards)
                        format_warning = (
                            "" if parsed_cards
                            else "Could not parse any Q:/A: card pairs. The "
                            "raw text is shown below; try generating again."
                        )
                        st.session_state["flashcard_pairs"] = parsed_cards
                    else:
                        exam_material, format_valid, format_warning = (
                            generate_exam_material(
                                question_type=question_type,
                                topic=topic,
                                marks=marks,
                                answer_style=answer_style,
                                difficulty=difficulty,
                                count=count,
                                context=context
                            )
                        )
                        st.session_state.pop("flashcard_pairs", None)

                    # A new generation invalidates any previous quiz
                    st.session_state.pop("quiz_questions", None)
                    st.session_state.pop("quiz_submitted", None)

                    st.session_state["exam_material"] = exam_material
                    st.session_state["exam_document"] = file_hash
                    st.session_state["exam_sources"] = passages
                    st.session_state["exam_distances"] = distances
                    st.session_state["exam_format"] = question_type
                    st.session_state["exam_format_valid"] = format_valid
                    st.session_state["exam_format_warning"] = format_warning

        except Exception as error:
            st.error(
                "Could not generate study material. Check that Ollama is "
                "running and the qwen2.5:3b and nomic-embed-text models are "
                f"available.\n\nDetails: {error}"
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

            is_flashcards = (
                st.session_state.get("exam_format") == "Flashcards (Q&A cards)"
                and bool(st.session_state.get("flashcard_pairs"))
            )

            if is_flashcards:
                for card_index, card in enumerate(
                    st.session_state["flashcard_pairs"], start=1
                ):
                    with st.expander(f"🎴 Card {card_index}: {card['question']}"):
                        st.markdown(card["answer"])
                with st.expander("View flashcards as plain text"):
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

            # ---------- Interactive MCQ quiz ----------
            if (
                st.session_state.get("exam_format")
                == "Multiple-choice questions (MCQs)"
            ):
                st.divider()
                if not st.session_state.get("quiz_questions"):
                    if st.button(
                        "🧩 Practice these MCQs as an interactive quiz",
                        key="start_quiz"
                    ):
                        parsed = parse_mcqs(exam_material)
                        if parsed:
                            st.session_state.quiz_questions = parsed
                            st.session_state.quiz_submitted = False
                            for q_index in range(len(parsed)):
                                st.session_state.pop(
                                    f"quiz_choice_{q_index}", None
                                )
                        else:
                            st.warning(
                                "The MCQs could not be parsed into an "
                                "interactive quiz. Please use the notes view "
                                "above, or try generating again."
                            )

                if st.session_state.get("quiz_questions"):
                    quiz_questions = st.session_state.quiz_questions
                    option_letters = ["A", "B", "C", "D"]

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
                    if col_reset.button("🔄 Reset quiz", key="reset_quiz"):
                        st.session_state.pop("quiz_questions", None)
                        st.session_state.pop("quiz_submitted", None)
                        for q_index in range(len(quiz_questions)):
                            st.session_state.pop(f"quiz_choice_{q_index}", None)

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
                            if item.get("explanation"):
                                st.caption(
                                    f"Explanation: {item['explanation']}"
                                )
                        st.markdown(
                            f"### 🏁 Score: {score} / {len(quiz_questions)}"
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