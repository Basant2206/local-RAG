import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Streamlit UI
st.set_page_config(page_title="Enterprise RAG Chat Assistant", layout="wide")
st.title("Local RAG Chat Assistant")
st.markdown("Powered by **Ollama** | 100% Private& Local | No API Key Required")

# Initialize session (Cached for performance optimization)
@st.cache_resource 
def chat_with_docs():
    # load existing local database
    embeddings=HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vector_db = Chroma(
        embedding_function = embeddings,
        persist_directory = "./vector_db"
    )

    # define llm
    #llm = OllamaLLM(model="gemma4")     # llama3.2, mistral
    llm = OllamaLLM(model="mistral")

    # Custom Prompt Template
    template = """ <thinking>
    Use the provided context to plan factual answer.
    </thinking>
    Context:{context}
    Question: {question}
    Answer.
    """

    qa_chain_prompt= PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm = llm,
        retriever = vector_db.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs = {"prompt": qa_chain_prompt},
        return_source_documents = True
    )

qa_chain =chat_with_docs()

# chat history management
if "history" not in st.session_state:
    st.session_state.history = []

for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
    

# User input
if prompt := st.chat_input("Ask a question about the document:"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": prompt})
            st.markdown(response['result'])
            st.session_state.history.append({"role": "assistant", "content": response['result']})
            sources = response.get("source_documents", [])
            if sources:
                st.markdown("**Sources:**")
                for doc in sources:
                    st.markdown(f"- {doc.metadata.get('source', 'Unknown Source')}")

            # Show Citation
            with st.expander("View verification details"):
                for i, doc in enumerate(sources):
                    src_name = doc.metadata.get('source', 'Unknown')
                    st.info(f"**Reference {i+1} \n\n {src_name}** : {doc.page_content[:200]}...")
                    
                    