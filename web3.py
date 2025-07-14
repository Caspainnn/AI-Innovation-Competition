import streamlit as st
import time
import requests
import jieba

# 初始化会话状态
if "start_chat" not in st.session_state:
    st.session_state.start_chat = False

# ======================= 首页 =======================
if not st.session_state.start_chat:
    # 美化容器
    with st.container():
        st.header("⚖️ 法务智能助手")
        st.info(
            "本项目结合了 **检索(Retrieval)**、**重排(Rerank)** 和 **生成(Generation)** "
            "技术，为您提供更精准的回答。"
        )

        st.subheader("为什么选择我们:")

        # 第一行，2 个列
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #1f4e79;">RAG检索增强</h3>
                <p style="color: #333;">采用检索增强生成技术，自动从法律文档中检索相关条文，确保回答的准确性和权威性，避免AI幻觉问题。</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #1f4e79;">完整法规库</h3>
                <p style="color: #333;">集成企业破产法、刑法、民事诉讼法、消费者权益保护法等核心法律法规，覆盖日常法律咨询的主要领域。</p>
            </div>
            """, unsafe_allow_html=True)

        # 第二行，3 个列
        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #1f4e79;">实时响应</h3>
                <p style="color: #333;">优化的向量化检索算法和本地化部署，确保快速响应，平均响应时间小于3秒，提供流畅的咨询体验。</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #1f4e79;">隐私保护</h3>
                <p style="color: #333;">本地化部署，数据不出域，严格保护用户隐私和咨询内容，符合数据安全和隐私保护的最高标准。</p>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="color: #1f4e79;">专业权威</h3>
                <p style="color: #333;">所有回答均基于官方法律条文，提供法条引用和出处，确保法律解释的专业性和权威性</p>
            </div>
            """, unsafe_allow_html=True)


    # 使用 st.button 替代自定义 HTML 按钮
    if st.button("Let's Start !", use_container_width=True):
        st.session_state.start_chat = True
        st.rerun()

if st.session_state.start_chat:
    # ======================= 咨询页面 =======================
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


    def stream_data(data):
        for word in jieba.cut(data):
            yield word
            time.sleep(0.03)  # 适当调整延迟时间


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

        # 修复：对话轮次统计应该按query计算，而不是按总条目数
        conversation_count = len(st.session_state.chat_history)
        # 如果正在处理新请求，显示即将到来的对话轮次
        if st.session_state.processing:
            conversation_count += 1
        st.info(f"🗨️ 对话轮次：{conversation_count}")

        MODEL_LIST = ["GLM_V4", "Qwen_32B", "DeepSeek_R1", "快速模式"]
        selected_model = st.selectbox("☑️ **选择模型**", MODEL_LIST)

    # 展示历史对话内容
    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(chat["query"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            render_reference_articles(chat.get("refs", []), chat["answer"])
            # 只有在不处理新请求时才显示思考时间，避免重复显示
            if "response_time" in chat and not st.session_state.processing:
                st.caption(f"⏱️ 思考时间：{chat['response_time']} 秒")

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
                    response = requests.post("http://127.0.0.1:8000/query", json={
                        "query": query,
                        "history": st.session_state.chat_history,
                        "model_name": selected_model})
                    response.raise_for_status()
                    result = response.json()
                    answer = result.get("answer", "❌ 未返回回答")
                    references = result.get("references", [])
                except requests.exceptions.RequestException as e:
                    answer = f"❌ 请求失败：{e}"
                    references = []

            # 响应时间
            response_time = time.time() - start_time

            # 🌟 流式输出回答
            final_answer = ""
            placeholder = st.empty()
            for chunk in stream_data(answer):
                final_answer += chunk
                placeholder.markdown(final_answer)

            # ✅ 显示参考法条
            render_reference_articles(references, final_answer)

            # ✅ 显示思考时间
            st.caption(f"⏱️ 思考时间：{response_time:.2f} 秒")

            # 修复：先保存到历史记录，再重置processing状态
            st.session_state.chat_history.append({
                "query": query,
                "answer": answer,
                "refs": references,
                "response_time": f"{response_time:.2f}"
            })

        # 重置处理状态
        st.session_state.processing = False
        # 立即刷新页面显示最新的对话轮次和避免重复显示
        st.rerun()
