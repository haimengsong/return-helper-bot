import streamlit as st

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from PyPDF2 import PdfReader

st.set_page_config(
        page_title="PDF QA ChatBot",
        page_icon=" ",
        layout="wide"
    )
st.title(" PDF问答机器人")

def set_api_key():
    user_api_key = st.sidebar.text_input(
        label="#### 在此填入API key填写完成后回车 ",
        placeholder="Paste your openAI API key, sk-",
        type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
        st.sidebar.write(" 填入成功下一步选取pdf")

def set_pdf_upload():
    # 在侧边栏设置标题
    st.sidebar.title("请选择PDF文件")

    # 在侧边栏创建一个文件上传器，限制文件类型为PDF，且一次只能上传一个文件
    global pdf_list
    pdf_list = st.sidebar.file_uploader("一次性选择一个PDF文件", type="pdf", accept_multiple_files=False)
    # 如果用户已经上传了文件，就在侧边栏显示一个消息
    if pdf_list is not None:
        initialize_db(pdf_list)
        st.sidebar.write("文件载入成功，现在可以进行文档问答")

    return pdf_list

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
def initialize_db(pdf_list):
    text = ""
    # 使用 PyPDF2 的 PdfReader 读取 pdf 文件
    pdf_reader = PdfReader(pdf_list)

    for page in pdf_reader.pages:
        text += page.extract_text()

    text_chunks = get_text_chunks(text)

    db = save_vector_store(text_chunks)
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)

    global PDF_QA
    PDF_QA = RetrievalQA.from_chain_type(llm,
                                                    retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                              search_kwargs={"score_threshold": 0.7}))
    PDF_QA.return_source_documents = True

    return PDF_QA

def load_vector_store():
    return FAISS.load_local('faiss', AzureOpenAIEmbeddings())

def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = PDF_QA({"query": message})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "I don't know."

def set_chat_ui():
    # 如果会话状态中没有"messages"这个键，就创建一个空列表
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask your question: "):
        if pdf_list is not None and os.environ["AZURE_OPENAI_API_KEY"] is not None:
            # 在会话状态的消息列表中添加用户的输入
            st.session_state.messages.append({"role": "user", "content": user_input})
            # 在聊天窗口中显示用户的输入
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = chat(user_input, st.session_state.messages)
                message_placeholder.markdown(full_response)
                # 在会话状态的消息列表中添加助手的回答
                st.session_state.messages.append({"role": "assistant", "content": full_response})

        if os.environ["AZURE_OPENAI_API_KEY"] is None:
            st.empty().markdown("请先输入您的API Key")

        if pdf_list is None:
            st.empty().markdown("请先上传您的PDF文件")

    if len(st.session_state.messages) > 20:
        st.session_state.messages = st.session_state.messages[-20]


if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"

    set_api_key()
    set_pdf_upload()
    set_chat_ui()