import ollama
import chromadb

# 1. Open the saved ChromaDB database
client = chromadb.PersistentClient(path="./studymate_db")

# 2. Open the Machine Learning notes collection
collection = client.get_or_create_collection(
    name="machine_learning_notes"
)

print("=" * 45)
print("          Welcome to StudyMate AI")
print("=" * 45)
print("Ask questions about your Machine Learning notes.")
print("Type 'exit' to quit.\n")
print("Stored chunks:", collection.count())

if collection.count() == 0:
    print("No chunks found. Run store_pdf.py first.")
    raise SystemExit

# 3. Start a question-answering loop
while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("StudyMate AI: Happy studying!")
        break

    if not question:
        continue

    # 4. Convert the question into an embedding
    result = ollama.embed(
        model="nomic-embed-text",
        input=question
    )
    question_vector = result["embeddings"][0]

    # 5. Retrieve the three closest passages from ChromaDB
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=3
    )

    passages = results["documents"][0]

    # 6. Combine the retrieved text into context
    context = "\n\n".join(
        f"Passage {i}: {passage}"
        for i, passage in enumerate(passages, start=1)
    )

    # 7. Ask Qwen to answer using the retrieved context
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": (
    "You are StudyMate AI, an academic assistant that answers "
    "questions using the supplied passages from the student's notes. "
    "Treat the passages as your only source of factual information. "
    "Do not use your general knowledge to add facts, examples, "
    "analogies, or explanations that are not supported by the passages. "
    "Do not invent examples. "
    "You may simplify or paraphrase information, but preserve its meaning. "
    "If the passages do not contain enough information to answer, "
    "say: 'The provided notes do not contain enough information "
    "to answer this question.'"
)
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved passages:\n{context}\n\n"
                    "Answer the question using the passages above."
                )
            }
        ]
    )

    # 8. Display the answer
    answer = response["message"]["content"]
    print("\nStudyMate AI:", answer)

    # 9. Display the sources used
    print("\n--- Retrieved sources ---")

    for i, passage in enumerate(passages, start=1):
        distance = results["distances"][0][i - 1]
        print(f"\nPassage {i} | Distance: {distance:.4f}")
        print(passage)

print("\nStudy session ended.")