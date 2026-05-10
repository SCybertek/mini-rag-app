#https://docs.langchain.com/langsmith/evaluation-quickstart?utm_source=chatgpt.com

from dotenv import load_dotenv
from langsmith import Client

from app import ask

load_dotenv()

client = Client()

DATASET_NAME = "mini-rag-eval-dataset"


# 1. Create dataset if it does not exist
try:
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    print(f"Using existing dataset: {DATASET_NAME}")
except Exception:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for mini RAG app"
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {
                    "question": "What does RAG stand for?"
                },
                "outputs": {
                    "expected_keywords": ["Retrieval", "Augmented", "Generation"]
                },
            },
            {
                "inputs": {
                    "question": "What does RAG help an AI system do?"
                },
                "outputs": {
                    "expected_keywords": ["external documents", "model memory"]
                },
            },
            {
                "inputs": {
                    "question": "Who won the Super Bowl?"
                },
                "outputs": {
                    "expected_keywords": ["I don't know"]
                },
            },
        ],
    )

    print(f"Created dataset: {DATASET_NAME}")


# 2. Target function: this is what LangSmith will test
def rag_target(inputs: dict) -> dict:
    answer = ask(inputs["question"])
    return {"answer": answer}


# 3. Evaluator: score answer based on expected keywords
# keyword match_evaluator checks how many of the expected keywords are present in the answer and gives a score between 0 and 1. (0.5 for half matched)
# It also returns a comment with details on which keywords were matched.
# it is brittle but simple

def keyword_match_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs["answer"].lower()
    expected_keywords = reference_outputs["expected_keywords"]

    matched = [
        keyword for keyword in expected_keywords
        if keyword.lower() in answer
    ]

    score = len(matched) / len(expected_keywords)

    return {
        "key": "keyword_match",
        "score": score,
        "comment": f"Matched {len(matched)} of {len(expected_keywords)} keywords: {matched}"
    }


# 4. Run evaluation
results = client.evaluate(
    rag_target,
    data=DATASET_NAME,
    evaluators=[keyword_match_evaluator],
    experiment_prefix="mini-rag-keyword-eval",
    blocking=True,
)

print("Evaluation completed.")
print("Check LangSmith Experiments for results.")