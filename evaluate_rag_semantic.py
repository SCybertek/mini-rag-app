from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app import ask

load_dotenv()

client = Client()

DATASET_NAME = "mini-rag-semantic-eval-dataset"
# This file is similar to evaluate_rag.py but instead of keyword matching, 
# it uses an LLM judge to evaluate the semantic correctness of the answer.
# in other words, it checks if the meaning of the answer is correct.

# LLM used as evaluator/judge
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 1. Create dataset
try:
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    print(f"Using existing dataset: {DATASET_NAME}")
except Exception:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Semantic evaluation dataset for mini RAG app"
    )

    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {
                    "question": "What does RAG stand for?"
                },
                "outputs": {
                    "expected_answer": "RAG stands for Retrieval Augmented Generation."
                },
            },
            {
                "inputs": {
                    "question": "What does RAG help an AI system do?"
                },
                "outputs": {
                    "expected_answer": "RAG helps an AI system answer using external documents instead of relying only on model memory."
                },
            },
            {
                "inputs": {
                    "question": "Who won the Super Bowl?"
                },
                "outputs": {
                    "expected_answer": "The assistant should say it does not know based on the provided documents."
                },
            },
        ],
    )

    print(f"Created dataset: {DATASET_NAME}")


# 2. Target function: your RAG app
def rag_target(inputs: dict) -> dict:
    answer = ask(inputs["question"])
    return {"answer": answer}


# 3. Semantic evaluator
def semantic_correctness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> dict:
    question = inputs["question"]
    actual_answer = outputs["answer"]
    expected_answer = reference_outputs["expected_answer"]

    prompt = ChatPromptTemplate.from_template("""
You are evaluating a RAG application's answer.

Question:
{question}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}

Score the actual answer:
- 1.0 = correct and semantically equivalent
- 0.5 = partially correct
- 0.0 = incorrect

Important:
- Do not require exact wording.
- Focus on meaning.
- Penalize hallucinations.
- Penalize answers that contradict the expected answer.

Return only one of these values:
1.0
0.5
0.0
""")

    messages = prompt.format_messages(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer
    )

    judge_response = judge_llm.invoke(messages)
    raw_score = judge_response.content.strip()

    try:
        score = float(raw_score)
    except ValueError:
        score = 0.0

    return {
        "key": "semantic_correctness",
        "score": score,
        "comment": f"Judge score: {raw_score}"
    }


# 4. Run evaluation
results = client.evaluate(
    rag_target,
    data=DATASET_NAME,
    evaluators=[semantic_correctness_evaluator],
    experiment_prefix="mini-rag-semantic-eval",
    blocking=True,
)

print("Semantic evaluation completed.")
print("Check LangSmith Experiments.")