import ollama
import chromadb

# 1. Open the existing database
client = chromadb.PersistentClient(path="./studymate_db")

# 2. Open the existing collection
collection = client.get_or_create_collection(
    name="machine_learning_notes"
)

print("Chunks stored:", collection.count())

# 3. Ask a question
question = input("\nAsk a question about your notes: ")

# 4. Convert the question into an embedding
result = ollama.embed(
    model="nomic-embed-text",
    input=question
)
question_vector = result["embeddings"][0]

# 5. Search the database
results = collection.query(
    query_embeddings=[question_vector],
    n_results=3
)

# 6. Display the retrieved passages
print("\n--- Retrieved passages ---")

for i, passage in enumerate(results["documents"][0], start=1):
    distance = results["distances"][0][i - 1]
    print(f"\nPassage {i} | Distance: {distance:.4f}")
    print(passage)