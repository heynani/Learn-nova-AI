import ollama

text = input("Enter a sentence: ")

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

vector = response["embeddings"][0]

print("\nEmbedding generated successfully!")
print("Number of values in the vector:", len(vector))
print("First 10 values:", vector[:10])