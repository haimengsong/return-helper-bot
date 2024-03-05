import streamlit as st

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

import os
from dotenv import load_dotenv

from PyPDF2 import PdfReader


def get_pdf_text(pdf_path):
    text = ""
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks


def save_vector_store(textChunks):
    db = FAISS.from_texts(textChunks, AzureOpenAIEmbeddings())
    db.save_local('faiss')
    return db


def load_vector_store():
    return FAISS.load_local('faiss', AzureOpenAIEmbeddings())


def initialize_data(str="IRM_Help.pdf"):
    raw_text = get_pdf_text(str)
    text_chunks = get_text_chunks(raw_text)

    db = save_vector_store(text_chunks)
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)

    global RETURN_HELPER_BOT
    RETURN_HELPER_BOT = RetrievalQA.from_chain_type(llm,
                                                    retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                              search_kwargs={"score_threshold": 0.7}))
    RETURN_HELPER_BOT.return_source_documents = True

    return RETURN_HELPER_BOT


def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = RETURN_HELPER_BOT({"query": message})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "I don't know."


def launch_ui():
    st.set_page_config(
        page_title="ReturnHelper",
        page_icon=" ",
        layout="wide"
    )
    st.title(" Return Helper ChatBot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask your question: "):
        with st.chat_message("user"):
            st.markdown(user_input)

    ai_response = chat(user_input, st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": user_input})

    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    if len(st.session_state.messages) > 20:
        st.session_state.messages = st.session_state.messages[-20]


if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    os.environ["AZURE_OPENAI_API_KEY"] = "8d1daadc333e42b18e26d861588cfd43"
    env_path = os.getenv("HOME") + "/Downloads/.env"
    load_dotenv(dotenv_path=env_path, verbose=True)

    initialize_data()

    launch_ui()