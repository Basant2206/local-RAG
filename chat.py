from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def chat_with_docs():
    # load existing local database
    embeddings=HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vector_db = Chroma(
        embedding_function = embeddings,
        persist_directory = "./vector_db"
    )

    # define llm
    llm = OllamaLLM(model="gemma4")     # llama3.2, mistral

    # Custom Prompt Template
    template = """ Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, say you don't know without try to make anwer.

    Context:{context}
    Question: {question}
    """

    qa_chain_prompt= PromptTemplate.from_template(template)

    # Set up retriver
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = vector_db.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs = {"prompt": qa_chain_prompt}
    )

    print("\n Local RAG chat Assistant is ready to chat! Type 'exit' to quit.\n")
    while True:
        query = input("Ask querry about document: ")
        if query.lower() == "exit":
            break
        response = qa_chain.invoke(query)
        print(f"\nAI: {response['result']}\n")

if __name__ == "__main__":
    chat_with_docs()