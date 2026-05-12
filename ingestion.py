from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_db(path):
    # load pdf
    loader = PyPDFLoader(path)
    raw_docs = loader.load()

    # load pdf, overlap to prevent context loss
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_docs) 
    
    # HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

    # save to local folder
    vector_db = Chroma.from_documents(
        documents = documents,
        embedding = embeddings,
        persist_directory = "./vector_db"
    )

    print(f"Knowledge Base Ready {len(documents)}")
      
    


    
if __name__ == "__main__":
    create_vector_db("Internship.pdf")
