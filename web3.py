import streamlit as st
import time
import requests

# 页面配置
st.set_page_config(page_title="法律智能助手", page_icon="⚖️", layout="centered")
st.title("⚖️ 法务智能助手")
st.caption("基于RAG技术的智能法务问答系统 | SiliconFlow AI引擎支持")


# 封装展示引用条文的函数
def render_references(reference_chunks, caption, is_current=False):
    st.caption(caption)
    if reference_chunks:
        # 只有当前回答才展开，历史对话都折叠
        expanded = is_current
        with st.expander("📚 查看引用条文", expanded=expanded):
            doc_groups = {}
            for doc in reference_chunks:
                name = doc.get("source", "未知来源").split("\\")[-1].split("/")[-1]
                preview = doc.get("preview", "⚠️ 无预览内容")
                doc_groups.setdefault(name, []).append(preview)
            for name, chunks in doc_groups.items():
                st.markdown(f"#### 📄 {name}")
                for idx, chunk in enumerate(chunks):
                    if len(chunks) > 1:
                        st.markdown(f"**片段 {idx + 1}:**")
                    st.markdown(chunk)
                    if idx < len(chunks) - 1:
                        st.markdown("---")


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
for idx, round in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(round["query"])
    with st.chat_message("assistant"):
        st.markdown(round["answer"])
        # 历史对话的引用条文默认折叠，显示历史响应时间
        render_references(
            round["references"],
            f"⏱️ 响应时间：{round.get('response_time', 'N/A')}秒 | 📑 引用条文数：{len(round['references'])}",
            is_current=False
        )

# 用户输入框
query = st.chat_input("请输入你的法律问题...")

# 处理新查询
if query and not st.session_state.processing:
    st.session_state.processing = True

    # 立即显示用户输入
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
                references = result.get("references", [])
            except requests.exceptions.RequestException as e:
                answer = f"❌ 请求失败：{e}"
                references = []

        # 计算响应时间
        response_time = time.time() - start_time

        # 显示回答
        st.markdown(answer)

        # 当前回答的引用条文默认展开，显示响应时间
        render_references(
            references,
            f"⏱️ 响应时间：{response_time:.2f}秒 | 📑 引用条文数：{len(references)}",
            is_current=True
        )

        # 保存会话到历史记录，包含响应时间
        st.session_state.chat_history.append({
            "query": query,
            "answer": answer,
            "references": references,
            "response_time": f"{response_time:.2f}"  # 保存格式化的响应时间
        })

    # 重置处理状态
    st.session_state.processing = False

    # 重新运行以刷新界面
    st.rerun()