# Mini RAG App

A small Retrieval-Augmented Generation (RAG) project built with LangChain, OpenAI models, Chroma, and LangSmith evaluation.

The app loads text files from `docs/`, splits them into chunks, embeds the chunks with OpenAI embeddings, retrieves the most relevant chunks for a question, and asks a chat model to answer using only that retrieved context.

## Project Structure

```text
.
├── app.py                    # Main RAG pipeline and CLI chat loop
├── docs/
│   └── rag_notes.txt         # Source document used by the retriever
├── evaluate_rag.py           # LangSmith keyword-based evaluation
├── evaluate_rag_semantic.py  # LangSmith LLM-judge semantic evaluation
└── test_questions.py         # Simple assertion-based smoke tests
```

## Requirements

- Python 3.10+
- An OpenAI API key
- Optional: LangSmith API settings if you want to run the evaluation scripts

The project does not currently include a dependency manifest. Install the packages imported by the code:

```bash
pip install python-dotenv langchain-community langchain-text-splitters langchain-openai langchain-chroma langchain-core langsmith
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install python-dotenv langchain-community langchain-text-splitters langchain-openai langchain-chroma langchain-core langsmith
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
```

For LangSmith evaluations, also add:

```bash
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=mini-rag-app
```

## Run the App

Start the interactive command-line app:

```bash
python app.py
```

Ask questions about the content in `docs/rag_notes.txt`. Type `exit` or `quit` to stop.

Example questions:

```text
What does RAG stand for?
What does RAG help an AI system do?
Who won the Super Bowl?
```

The app is instructed to answer only from the retrieved document context. If the answer is missing from the documents, it should respond:

```text
I don't know based on the provided documents.
```

## Run Smoke Tests

```bash
python test_questions.py
```

These tests call `ask()` directly and assert that expected phrases appear in the answers.

## Run Evaluations

Keyword-based LangSmith evaluation:

```bash
python evaluate_rag.py
```

Semantic LLM-judge LangSmith evaluation:

```bash
python evaluate_rag_semantic.py
```

Both scripts create or reuse a LangSmith dataset, run the RAG app against the dataset examples, and publish experiment results to LangSmith.

## How It Works

1. `DirectoryLoader` loads `.txt` files from `docs/`.
2. `RecursiveCharacterTextSplitter` splits documents into overlapping chunks.
3. `OpenAIEmbeddings` embeds chunks with `text-embedding-3-small`.
4. `Chroma` stores the embedded chunks in an in-memory vector store.
5. The retriever returns the top 4 relevant chunks for each question.
6. `ChatOpenAI` uses `gpt-4o-mini` to answer from the retrieved context.

## Notes

- `app.py` rebuilds the vector store each time it runs.
- The Chroma store is currently in memory. The `persist_directory` option is commented out.
- Add more `.txt` files under `docs/` to expand the knowledge base.
- Because `app.py` initializes the RAG pipeline at import time, scripts that import `ask` will also load documents and create embeddings.
