import streamlit as st
import time
import requests

# 页面配置
st.set_page_config(page_title="法律智能助手", page_icon="⚖️", layout="centered")
st.title("⚖️ 法务智能助手")
st.caption("基于RAG技术的智能法务问答系统 | SiliconFlow AI引擎支持")

# 自定义展示参考法律条文的函数
def render_reference_articles(refs, answer_text):
    if refs and "未在文档中找到明确依据" not in answer_text:
        with st.expander("📚 参考法律条文", expanded=False):
            source_groups = {}

            for doc in refs:
                # 提取来源文件名（你也可以换成结构化的法典名称）
                source_path = doc.get("source", "未知来源")
                source_name = source_path.split("\\")[-1].split("/")[-1]
                preview = doc.get("preview", "⚠️ 无预览内容")
                score = float(doc.get("score", 0.0))
                source_groups.setdefault(source_name, []).append({
                    "preview": preview.strip(),
                    "score": score
                })

            # 展示每个唯一来源文件一次
            for source_name, docs in source_groups.items():
                st.caption(f"📄 来源文件名：{source_name}")
                for i, entry in enumerate(docs):
                    st.markdown(f"**文档 {i + 1}** (相关性评分: {entry['score']:.4f})")
                    st.code(entry["preview"])



# 初始化状态变量
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Sidebar：系统信息
with st.sidebar:
    st.header("📋 系统信息")
    st.subheader("🔗 API 信息")
    st.markdown("- **服务端口**: http://127.0.0.1:8000")
    st.header("📋 系统状态")
    try:
        health_response = requests.get("http://127.0.0.1:8000/", timeout=2)
        if health_response.status_code == 200:
            st.success("✅ 后端服务正常")
        else:
            st.error("❌ 后端服务异常")
    except:
        st.error("❌ 后端服务连接失败")
    st.info(f"🗨️ 对话轮次：{len(st.session_state.chat_history)}")

# 展示历史对话内容
for idx, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(chat["query"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
        render_reference_articles(chat.get("refs", []),chat["answer"])

# 用户输入框
query = st.chat_input("请输入你的法律问题...")

# 处理新查询
if query and not st.session_state.processing:
    st.session_state.processing = True

    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(query)

    # 开始计时
    start_time = time.time()

    with st.chat_message("assistant"):
        with st.spinner("正在思考中..."):
            try:
                response = requests.post("http://127.0.0.1:8000/query", json={"query": query})
                response.raise_for_status()
                result = response.json()
                answer = result.get("answer", "❌ 未返回回答")
                references = result.get("references", [])  # references = List[Tuple[Dict, float]]
            except requests.exceptions.RequestException as e:
                answer = f"❌ 请求失败：{e}"
                references = []

        # 响应时间
        response_time = time.time() - start_time

        # 显示回答内容
        st.markdown(answer)
        render_reference_articles(references, answer)

        # 保存历史
        st.session_state.chat_history.append({
            "query": query,
            "answer": answer,
            "refs": references,
            "response_time": f"{response_time:.2f}"
        })

    # 重置处理状态
    st.session_state.processing = False
    st.rerun()
