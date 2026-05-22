import os
import re
import base64
from typing import Any, List

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from rank_bm25 import BM25Okapi
from pydantic import Field

load_dotenv()

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


# ── クエリ書き換えプロンプト ──────────────────────────────────────────────────
QUERY_REWRITE_PROMPT = """あなたはベーカリーの情報を検索するためのクエリ生成AIです。
ユーザーの質問に対して、ナレッジベースから必要な情報を
確実に取得するための検索クエリを3つ生成してください。

【生成するクエリの方向】
1. 元の質問の言い換え（別の表現で同じことを聞く）
2. 質問の分解（複数の概念が含まれる場合、個別に分ける）
3. 具体化（あいまいな表現を具体的な用語に変換する）

【出力形式】
クエリ1:（テキスト）
クエリ2:（テキスト）
クエリ3:（テキスト）

それ以外の文章は一切出力しないこと。"""


# ── 日本語トークナイザー（bigram + unigram）────────────────────────────────
def _tokenize(text: str) -> List[str]:
    text = re.sub(r'[#*\-\|「」【】（）・\s]+', '', text)
    if not text:
        return ['']
    tokens: List[str] = []
    for i in range(len(text)):
        tokens.append(text[i])
        if i < len(text) - 1:
            tokens.append(text[i:i + 2])
    return tokens


# ── ハイブリッド検索 Retriever ────────────────────────────────────────────
class HybridMultiQueryRetriever(BaseRetriever):
    """クエリ書き換え + BM25 + ChromaDB + RRF 統合 Retriever"""

    vectorstore: Any = Field(...)
    documents: List[Any] = Field(...)
    bm25: Any = Field(...)
    llm: Any = Field(...)
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True

    def _generate_queries(self, question: str) -> List[str]:
        try:
            response = self.llm.invoke([
                SystemMessage(content=QUERY_REWRITE_PROMPT),
                HumanMessage(content=question),
            ])
            queries: List[str] = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if re.match(r'^クエリ\d+[:：]', line):
                    q = re.sub(r'^クエリ\d+[:：]\s*', '', line).strip('（）() ')
                    if q:
                        queries.append(q)
            return queries[:3] if queries else [question]
        except Exception:
            return [question]

    def _vector_search(self, query: str, k: int = 5) -> List[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def _bm25_search(self, query: str, k: int = 5) -> List[str]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i].page_content for i in top_k]

    def _rrf(self, rankings: List[List[str]], k_rrf: int = 60) -> List[str]:
        scores: dict = {}
        for ranking in rankings:
            for rank, content in enumerate(ranking):
                scores[content] = scores.get(content, 0.0) + 1.0 / (k_rrf + rank + 1)
        return sorted(scores, key=lambda x: scores[x], reverse=True)[:self.k]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        queries = self._generate_queries(query)

        rankings: List[List[str]] = []
        for q in queries:
            rankings.append(self._vector_search(q, k=5))
            rankings.append(self._bm25_search(q, k=5))

        top_contents = self._rrf(rankings)

        content_map = {doc.page_content: doc for doc in self.documents}
        return [
            content_map.get(c, Document(page_content=c))
            for c in top_contents
        ]


# ── チャンク構築（BM25 / ベクトルDB 共通）────────────────────────────────
def _build_chunks() -> List[Document]:
    headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    chunks: List[Document] = []
    for fname in ["knowledge/products.md", "knowledge/store_info.md", "knowledge/allergens.md"]:
        docs = TextLoader(fname, encoding="utf-8").load()
        for doc in docs:
            chunks.extend(splitter.split_text(doc.page_content))
    return chunks


# ── チェーン構築（キャッシュ）────────────────────────────────────────────
@st.cache_resource
def load_chain():
    chunks = _build_chunks()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists("chroma_db"):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db",
        )
    else:
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings,
        )

    tokenized = [_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = HybridMultiQueryRetriever(
        vectorstore=vectorstore,
        documents=chunks,
        bm25=bm25,
        llm=llm,
        k=5,
    )

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


# ── Streamlit UI ──────────────────────────────────────────────────────────
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
