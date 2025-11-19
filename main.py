import os
from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def build_vector_store():
    print("Loading speech.txt ...")
    loader = TextLoader("speech.txt")
    documents = loader.load()

    print("Splitting text into chunks ...")
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    print("Creating embeddings using MiniLM-L6-v2 ...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Storing embeddings in ChromaDB ...")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    print("Vector DB created and saved!")

    return vectordb


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="db", embedding_function=embeddings)
    return vectordb


def chat():
    print("\nAmbedkarGPT — Ask anything based on the speech\n")

    if not os.path.exists("db"):
        vectordb = build_vector_store()
    else:
        vectordb = load_vector_store()

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="mistral")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False
    )

    while True:
        query = input("\nYour Question (or type 'exit'): ")
        if query.lower() == "exit":
            print("Exiting AmbedkarGPT.")
            break

        print("\nThinking...")
        answer = qa.run(query)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    chat()
