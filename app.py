import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. Load documents from the "docs" directory 
loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(documents)

# 3. Convert chunks into embeddings and store in vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    # persist_directory="./chroma_db"
    # persistence was causing duplicate embeddings.  
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. Create prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer only using the context below.
If the answer is not in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 5. Ask question
def ask(question, show_sources=False):
    retrieved_docs = retriever.invoke(question)

    if show_sources:
        print("\nRetrieved chunks:")
        for i, doc in enumerate(retrieved_docs, start=1):
            print(f"\n--- Chunk {i} ---")
            print("Source:", doc.metadata.get("source"))
            print(doc.page_content)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    try:
        while True:
            question = input("\nAsk a question: ")

            if question.lower() in ["exit", "quit"]:
                print("Goodbye 👋")
                break

            answer = ask(question, show_sources=True) # show_sources is set to True for demonstration. You can set it to False if you don't want to see the retrieved chunks.
            print("\nAnswer:")
            print(answer)

    except KeyboardInterrupt:
        print("\nStopped by user. Goodbye 👋")