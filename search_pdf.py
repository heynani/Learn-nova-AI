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

# 2. Split the text into chunks
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

# 3. Generate embeddings for all chunks
result = ollama.embed(
    model="nomic-embed-text",
    input=chunks
)
embeddings = result["embeddings"]

print(f"\nPDF loaded: {len(chunks)} chunks ready.")

# 4. Ask the student a question
question = input("\nAsk a question about your notes: ")

# 5. Generate an embedding for the question
question_result = ollama.embed(
    model="nomic-embed-text",
    input=question
)
question_vector = question_result["embeddings"][0]

# 6. Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)

# 7. Compare the question with every chunk
scores = []

for i, embedding in enumerate(embeddings):
    score = cosine_similarity(question_vector, embedding)
    scores.append((score, i))

# 8. Retrieve the three most relevant chunks
scores.sort(reverse=True)
top_results = scores[:3]

print("\n--- Most relevant passages from your PDF ---")

for rank, (score, index) in enumerate(top_results, start=1):
    print(f"\nResult {rank} | Similarity: {score:.4f}")
    print(chunks[index])