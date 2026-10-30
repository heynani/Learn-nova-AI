import pymupdf
import ollama
import math

# 1. Read the PDF
pdf_path = input("Enter your Machine Learning PDF path: ").strip().strip('"')
document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

# 2. Split the PDF text into chunks
words = full_text.split()
chunk_size = 100
overlap = 20
chunks = []
start = 0

while start < len(words):
    chunk = " ".join(words[start:start + chunk_size])
    chunks.append(chunk)
    start += chunk_size - overlap

if not chunks or not full_text.strip():
    print("No extractable text found in this PDF.")
    raise SystemExit

print(f"Loaded {len(chunks)} chunks from your PDF.")

# 3. Generate embeddings for the chunks
result = ollama.embed(
    model="nomic-embed-text",
    input=chunks
)
embeddings = result["embeddings"]

# 4. Define cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    mag_a = math.sqrt(sum(a * a for a in vector_a))
    mag_b = math.sqrt(sum(b * b for b in vector_b))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)

# 5. Ask the student a question
question = input("\nAsk a question about your notes: ")

# 6. Embed the question
question_result = ollama.embed(
    model="nomic-embed-text",
    input=question
)
question_vector = question_result["embeddings"][0]

# 7. Find the three most relevant chunks
scores = []

for index, embedding in enumerate(embeddings):
    score = cosine_similarity(question_vector, embedding)
    scores.append((score, index))

scores.sort(reverse=True)
top_results = scores[:2]

# 8. Prepare the retrieved text
context = ""

for rank, (score, index) in enumerate(top_results, start=1):
    context += f"\nPassage {rank}:\n{chunks[index]}\n"

# 9. Ask Qwen to answer using the retrieved text
response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "system",
           "content": (
    "You are StudyMate AI, a careful academic assistant. "
    "Answer using only information supported by the supplied passages. "
    "Explain in simple English. "
    "Do not add outside facts, invented examples, or unsupported details. "
    "If the passages do not contain enough information, "
    "say that the notes do not provide enough information. "
    "Do not claim that information is in the notes unless the passages support it."
)
        },
        {
            "role": "user",
            "content": (
                f"Student's question:\n{question}\n\n"
                f"Retrieved passages from the PDF:\n{context}\n\n"
                "Answer the question based on these passages."
            )
        }
    ]
)

# 10. Display the answer
print("\nStudyMate AI:", response["message"]["content"])

print("\n--- Retrieved passages ---")

for rank, (score, index) in enumerate(top_results, start=1):
    print(f"\nPassage {rank} | Similarity: {score:.4f}")
    print(chunks[index])