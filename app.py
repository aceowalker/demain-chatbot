import os
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import gradio as gr

load_dotenv()

# --- ベクトルDB読み込み ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# --- LLM設定 ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# --- メモリ（会話履歴保持） ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# --- RAGチェーン ---
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
)

def ask(question: str) -> str:
    result = qa_chain.invoke({"question": question})
    return result["answer"]

# --- 背景画像をBase64エンコード ---
def get_base64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image("assets/rogo002.jpg")

# --- JSでhtml/body/コンテナをすべて透明化し、背景をhtmlに設定 ---
js_fn = f"""
function() {{
    var dataUrl = 'data:image/jpeg;base64,{bg_image}';

    function applyBg() {{
        // html要素に背景を設定
        var html = document.documentElement;
        html.style.setProperty('background-image', 'url("' + dataUrl + '")', 'important');
        html.style.setProperty('background-size', 'cover', 'important');
        html.style.setProperty('background-position', 'center', 'important');
        html.style.setProperty('background-attachment', 'fixed', 'important');
        html.style.setProperty('min-height', '100vh', 'important');

        // body・Gradioコンテナをすべて透明化
        var selectors = [
            'body', '#root', 'gradio-app', '.gradio-container', '.main', '.contain',
            'footer', '.app', '.wrap', '.center', '.full'
        ];
        selectors.forEach(function(sel) {{
            document.querySelectorAll(sel).forEach(function(el) {{
                el.style.setProperty('background', 'transparent', 'important');
                el.style.setProperty('background-color', 'transparent', 'important');
                el.style.setProperty('background-image', 'none', 'important');
            }});
        }});

        // すべての要素を走査してwhiteな要素を透明化（#chat-wrapは除外）
        document.querySelectorAll('body *').forEach(function(el) {{
            if (el.closest('#chat-wrap')) return;
            var bg = window.getComputedStyle(el).backgroundColor;
            if (bg === 'rgb(255, 255, 255)' || bg === 'rgba(255, 255, 255, 1)') {{
                el.style.setProperty('background-color', 'transparent', 'important');
                el.style.setProperty('background', 'transparent', 'important');
            }}
        }});
    }}

    // 即時適用
    applyBg();

    // Svelteの再レンダリングに対応するため、5秒間は100ms毎に再適用
    var count = 0;
    var interval = setInterval(function() {{
        applyBg();
        count++;
        if (count >= 50) clearInterval(interval);
    }}, 100);
}}
"""

# --- コンポーネント用CSS ---
custom_css = """
html {
    min-height: 100vh;
}

body, .gradio-container, .main, .contain, footer {
    background: transparent !important;
    background-color: transparent !important;
}

#chat-wrap {
    max-width: 680px;
    margin: 40px auto;
    background: rgba(255, 255, 255, 0.88) !important;
    border-radius: 20px;
    padding: 32px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.35);
    backdrop-filter: blur(6px);
}

#chat-title {
    text-align: center;
    font-size: 1.6rem;
    color: #7a4500;
    margin-bottom: 4px;
    font-weight: 700;
    letter-spacing: 0.04em;
}

#chat-subtitle {
    text-align: center;
    font-size: 0.85rem;
    color: #b07020;
    margin-bottom: 20px;
    letter-spacing: 0.08em;
}

#send-btn {
    background: #E8890C !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
}

#send-btn:hover {
    background: #c97000 !important;
}

#clear-btn {
    background: transparent !important;
    color: #b07020 !important;
    border: 1px solid #b07020 !important;
    border-radius: 10px !important;
}
"""

# --- チャット関数 ---
def chat(message: str, history: list) -> tuple:
    response = ask(message)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return "", history

# --- Gradio UI ---
with gr.Blocks(css=custom_css, js=js_fn, title="石窯パン工房ドゥマン チャットボット") as demo:
    with gr.Column(elem_id="chat-wrap"):
        gr.HTML('<div id="chat-title">石窯パン工房 Demain</div>')
        gr.HTML('<div id="chat-subtitle">ご質問はこちらからどうぞ｜What can I help you with?</div>')

        chatbot = gr.Chatbot(
            label="",
            height=420,
            type="messages",
        )
        with gr.Row():
            msg = gr.Textbox(
                placeholder="例）塩パンの値段は？　定休日はいつ？　アレルギー情報を教えて",
                show_label=False,
                scale=5,
            )
            send_btn = gr.Button("送信", elem_id="send-btn", scale=1)

        clear_btn = gr.Button("会話をリセット", elem_id="clear-btn")

    # イベント
    send_btn.click(chat, [msg, chatbot], [msg, chatbot])
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(share=False, allowed_paths=["assets"])
