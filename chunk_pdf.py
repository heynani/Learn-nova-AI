import pymupdf

pdf_path = input("Enter the PDF file path: ")

document = pymupdf.open(pdf_path)

full_text = ""

for page in document:
    full_text += page.get_text() + "\n"

document.close()

words = full_text.split()

chunk_size = 100
overlap = 20

chunks = []
start = 0

while start < len(words):
    end = start + chunk_size
    chunk_words = words[start:end]

    chunks.append(" ".join(chunk_words))

    start += chunk_size - overlap

print(f"\nTotal words: {len(words)}")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)