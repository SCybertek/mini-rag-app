from app import ask

test_cases = [
    {
        "question": "What does RAG stand for?",
        "must_include": ["Retrieval", "Augmented", "Generation"]
    },
    {
        "question": "What should the assistant do if answer is missing?",
        "must_include": ["I don't know"]
    }
]

for test in test_cases:
    answer = ask(test["question"])
    print("\nQuestion:", test["question"])
    print("Answer:", answer)

    for phrase in test["must_include"]:
        assert phrase.lower() in answer.lower(), f"Missing: {phrase}"

print("\nAll tests passed.")