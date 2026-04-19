import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# --- ドキュメント読み込み ---
def load_markdown(filepath: str):
    loader = TextLoader(filepath, encoding="utf-8")
    return loader.load()

docs_products  = load_markdown("knowledge/products.md")
docs_store     = load_markdown("knowledge/store_info.md")
docs_allergens = load_markdown("knowledge/allergens.md")
all_docs = docs_products + docs_store + docs_allergens

# --- チャンク分割（Markdownヘッダー基準） ---
headers_to_split = [
    ("#",  "H1"),
    ("##", "H2"),
    ("###","H3"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)

chunks = []
for doc in all_docs:
    chunks.extend(splitter.split_text(doc.page_content))

print(f"チャンク数: {len(chunks)}")

# --- ベクトルDB構築・保存 ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("ベクトルDBの構築が完了しました。")
