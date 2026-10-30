import pymupdf
import ollama
import chromadb

# 1. Connect to a local, persistent ChromaDB database
client = chromadb.PersistentClient(path="./studymate_db")

# 2. Create or open a collection
collection = client.get_or_create_collection(
    name="machine_learning_notes"
)

# 3. Open the PDF
pdf_path = input("Enter your Machine Learning PDF path: ").strip().strip('"')
document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

# 4. Split the text into chunks
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

print(f"Created {len(chunks)} chunks.")

# 5. Generate embeddings and store them in batches
batch_size = 32

for start in range(0, len(chunks), batch_size):
    batch = chunks[start:start + batch_size]

    result = ollama.embed(
        model="nomic-embed-text",
        input=batch
    )

    embeddings = result["embeddings"]

    ids = [
        f"ml_chunk_{i}"
        for i in range(start, start + len(batch))
    ]

    collection.upsert(
        ids=ids,
        documents=batch,
        embeddings=embeddings,
        metadatas=[
            {"source": "machine_learning_notes.pdf"}
            for _ in batch
        ]
    )

print("PDF chunks and embeddings stored successfully!")
print("Total chunks in database:", collection.count())