import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Streamlit Cloud の secrets から APIキーを取得（ローカルは .env を優先）
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="石窯パン工房 Demain チャットボット", page_icon="🍞")


def get_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


bg_image = get_base64_image("assets/rogo002.jpg")

st.markdown(f"""
<style>
/* 背景画像 */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background-image: url("data:image/jpeg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
[data-testid="stHeader"] {{
    background: transparent;
}}

/* メインコンテナを半透明ウィンドウ化 */
[data-testid="stMainBlockContainer"] {{
    max-width: 720px;
    margin: 40px auto;
    background: rgba(255, 255, 255, 0.92) !important;
    border-radius: 20px;
    padding: 32px 40px 40px 40px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
    backdrop-filter: blur(8px);
    color: #333333 !important;
}}

/* チャットメッセージ背景を透明に・全テキストを強制的に暗色に */
[data-testid="stChatMessage"] {{
    background: transparent !important;
    color: #333333 !important;
}}
[data-testid="stChatMessage"] *,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] *,
[data-testid="stMarkdownContainer"] p {{
    color: #333333 !important;
}}

/* メインコンテナ内の全テキストを強制的に暗色に */
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] span,
[data-testid="stMainBlockContainer"] div,
[data-testid="stMainBlockContainer"] label {{
    color: #333333 !important;
}}

/* 入力欄 */
[data-testid="stChatInput"] textarea {{
    background: rgba(255,255,255,0.95) !important;
    color: #333333 !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: #888888 !important;
    opacity: 1 !important;
}}

#chat-title {{
    text-align: center;
    font-size: 1.6rem;
    color: #7a4500;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}}
#chat-subtitle {{
    text-align: center;
    font-size: 0.85rem;
    color: #b07020;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}}
div.stButton > button {{
    background-color: #E8890C;
    color: white;
    border-radius: 10px;
    font-weight: 700;
    border: none;
    width: 100%;
}}
div.stButton > button:hover {{
    background-color: #c97000;
    color: white;
}}
</style>
<div id="chat-title">石窯パン工房 Demain</div>
<div id="chat-subtitle">ご質問はこちらからどうぞ｜What can I help you with?</div>
""", unsafe_allow_html=True)


def build_vectorstore_if_needed():
    if not os.path.exists("chroma_db"):
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import MarkdownHeaderTextSplitter
        headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        chunks = []
        for fname in ["knowledge/products.md", "knowledge/store_info.md", "knowledge/allergens.md"]:
            docs = TextLoader(fname, encoding="utf-8").load()
            for doc in docs:
                chunks.extend(splitter.split_text(doc.page_content))
        Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chroma_db")


@st.cache_resource
def load_chain():
    build_vectorstore_if_needed()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
    )


qa_chain = load_chain()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("例）塩パンの値段は？　定休日はいつ？　アレルギー情報を教えて"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            result = qa_chain.invoke({"question": prompt})
            answer = result["answer"]
        st.write(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})

if st.session_state.history:
    if st.button("会話をリセット"):
        st.session_state.history = []
        qa_chain.memory.clear()
        st.rerun()
