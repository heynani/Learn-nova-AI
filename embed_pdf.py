import pymupdf
import ollama

# 1. Open the PDF
pdf_path = input("Enter your Machine Learning PDF path: ").strip().strip('"')
document = pymupdf.open(pdf_path)

# 2. Extract all text
full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

# 3. Split text into chunks
words = full_text.split()

chunk_size = 100
overlap = 20
chunks = []

start = 0

while start < len(words):
    chunk_words = words[start:start + chunk_size]
    chunks.append(" ".join(chunk_words))
    start += chunk_size - overlap

print("Total words:", len(words))
print("Total chunks:", len(chunks))

# 4. Generate an embedding for every chunk
if chunks:
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunks
    )

    embeddings = response["embeddings"]

    print("Total embeddings:", len(embeddings))
    print("First chunk:", chunks[0][:200])
    print("First 10 vector values:", embeddings[0][:10])
else:
    print("No text found in the PDF.")