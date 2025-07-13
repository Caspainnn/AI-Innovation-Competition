import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import json
import requests

os.environ['HF_HOME'] = 'D:/my_models_cache'

# 检查CUDA是否可用，并设置计算设备
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA is available! Using GPU for computation.")
else:
    device = "cpu"
    print("CUDA is NOT available. Using CPU for computation.")

from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader , Docx2txtLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# --- 配置 ---
FOLDER_PATH = "E:/self-cultivation/2025小学期/作业/大实验/sx/数据"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
FAISS_DB_PATH = "./faiss_index"
METADATA_FILE_NAME = "documents_metadata.json"

# --- 新增 SiliconFlow Reranker 配置 ---
SILICONFLOW_API_KEY =  os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" # 用户提供的模型名称

API_KEY = "sk-uyysjarbjuecjrgxacqnbbdrulbjlrtlsymfnkazrfinilee"

# 1. 加载文件夹中的数据
def load_documents_from_folder(folder_path:str):
    documents = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在：{folder_path}")

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path,file)
        try:
            if file.lower().endswith('.pdf'):
                # 加载PDF文件
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                print(f"✓ 已加载PDF文件: {file}")

            elif file.lower().endswith('.docx'):
                # 加载DOCX文件
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
                print(f"✓ 已加载DOCX文件: {file}")

        except Exception as e:
            print(f"× 加载失败 {file}: {str(e)}")
            continue
    print("📁 文档加载成功！")
    return documents

# 2. 对文档进行切块
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
        length_function = len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 切分后得到 {len(chunks)} 个文本块")
    return chunks

# 3. 加载嵌入模型
def get_embeddings_model(model_name_or_path: str) -> HuggingFaceEmbeddings:
    print(f"\n正在加载嵌入模型: {model_name_or_path} (device: {device})")
    # 这里确保模型会下载到HF_HOME指定目录，并使用指定设备
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_or_path,
        model_kwargs={'device': device}
    )
    print("✓ 嵌入模型加载完成。")
    return embeddings

# 4. 创建 FAISS 向量数据库并保存到本地
def create_and_save_faiss_db(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings, db_path: str) -> FAISS:
    print("\n正在创建 FAISS 向量数据库...")
    faiss_db = FAISS.from_documents(chunks, embeddings_model)
    print(f"✓ FAISS 向量数据库创建完成。正在保存到: {db_path}")
    faiss_db.save_local(db_path)
    print("🔧 FAISS 向量数据库保存成功。")
    return faiss_db

# 5. 创建并保存文档块的元数据
def create_and_save_metadata(chunks: list[Document], output_dir: str, metadata_file_name: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_list = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": f"chunk_{i}",
            "page_content_preview": chunk.page_content[:200] + "...", # 预览前200字符
            "source": chunk.metadata.get('source', '未知文件'),
            "page": chunk.metadata.get('page', '未知页码'),
            "start_index": chunk.metadata.get('start_index', '未知索引')
        }
        metadata_list.append(metadata)

    metadata_file_path = os.path.join(output_dir, metadata_file_name)
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
    print(f"\n元数据已保存到：{metadata_file_path}")

# --- SiliconFlow Reranker 函数 ---
def rerank_documents_siliconflow(
    query: str,
    documents: list[Document], # 接收 LangChain Document 对象列表
    model: str = RERANKER_MODEL,
    api_key: str = SILICONFLOW_API_KEY,
    top_n: int = 5 # 重排后返回的顶部文档数量
) -> list[Document]:
    # 使用 SiliconFlow Rerank API 对文档进行重排
    if not api_key:
        print("错误：未设置 SILICONFLOW_API_KEY 环境变量，无法调用 Reranker API。")
        return []

    doc_contents = [doc.page_content for doc in documents] # 提取文档内容发送给API

    payload = {
        "model": model,
        "query": query,
        "documents": doc_contents
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"\n正在调用 SiliconFlow Reranker ({model})...")
    try:
        response = requests.post(SILICONFLOW_RERANK_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status() # 对 4xx 或 5xx HTTP 状态码抛出异常
        result = response.json()

        if "results" not in result:
            print(f"Reranker API 响应格式不正确: {result}")
            return []

        # 将 rerank 结果与原始 Document 对象关联并排序
        reranked_items = []
        # SiliconFlow API 的 results 数组顺序与传入的 documents 数组顺序一致
        for i, res in enumerate(result["results"]):
            # 找到对应的原始 Document 对象。这里假设 res['document'] 直接是原始内容
            # 更严谨的话，可以在传入 documents 时给 Document 对象添加唯一 ID，然后通过 ID 匹配
            # 为了简化，我们假设 res['document'] 与 doc_contents[i] 相同
            reranked_items.append({
                "document": documents[i], # 原始 LangChain Document 对象
                "score": res['relevance_score']
            })

        # 按相关性分数降序排序
        reranked_items.sort(key=lambda x: x['score'], reverse=True)

        # 提取 top_n 文档
        top_reranked_docs = [item['document'] for item in reranked_items[:top_n]]

        # print(f"\nRerank 成功，返回 Top {len(top_reranked_docs)} 文档。\n")
        # for i, doc_item in enumerate(reranked_items[:top_n]):
        #     print(f"Top {i+1} (Score: {doc_item['score']:.4f}): {doc_item['document'].page_content[:100]}...")

        return top_reranked_docs

    except requests.exceptions.RequestException as e:
        print(f"调用 SiliconFlow Reranker API 失败: 网络或请求错误 - {e}")
        return []
    except json.JSONDecodeError:
        print(f"Reranker API 响应不是有效的 JSON: {response.text}")
        return []
    except Exception as e:
        print(f"Reranker 发生未知错误: {e}")
        return []

# 6. LLM 生成答案
def generate_llm_response(query: str, top_documents: list[Document]) -> str:
    """
        使用 LLM 对 top N 文档内容进行总结、合并，并生成结构化答案
    """
    context_text = "\n\n".join(
        f"段落{i + 1}：{doc.page_content}" for i, doc in enumerate(top_documents)
    )
    prompt = f"""
        你是一个精通中国法律的AI助手，现在有用户向你提出法律问题，请根据以下检索到的文档内容进行总结、归纳，输出权威且简洁的法律分析结果。
    
        用户问题：
        {query}
    
        相关参考文档：
        {context_text}
    
        请根据上述内容生成一段法律分析说明，要求如下：
        1. 语言客观、专业、清晰；
        2. 不要引用原文段落编号，整合成流畅文本；
        3. 若有多个方面，请用要点列举（如 "①...②..."）；
        4. 若没有足够信息，请说明“无法准确判断”。
    
        最终输出：
    """
    try:
        llm = ChatOpenAI(
            model="Qwen/QwQ-32B",
            api_key= API_KEY,
            base_url="https://api.siliconflow.cn/v1",
            temperature=0.3
        )

        messages = [
            {"role": "system", "content": "你是一个精通中国法律的法律助手，善于归纳总结并解释法律问题。"},
            {"role": "user", "content": prompt}
        ]

        response = llm.invoke(messages)

        # 根据实际返回结构判断是 content 字段还是 json 格式
        try:
            result = json.loads(response.content)
            return result.get("output", str(result))  # 支持多种格式
        except Exception:
            return response.content if isinstance(response.content, str) else str(response.content)

    except Exception as e:
        print(f"⚠️ LLM 总结失败：{e}")
        return "❌ 无法生成法律分析，请检查API密钥或网络连接。"

if __name__ == "__main__":
    # 1. 加载文档
    docs = load_documents_from_folder(FOLDER_PATH)
    print(f"📑 共加载 {len(docs)} 个文档页")

    # 2. 文本切块
    chunks_docs = split_documents(docs)

    # 3. 初始化嵌入向量
    embeddings = get_embeddings_model(EMBEDDING_MODEL)

    # 4. 创建并保存 FAISS 向量数据库
    faiss_db = create_and_save_faiss_db(chunks_docs, embeddings, FAISS_DB_PATH)

    # 5. 创建并保存元数据文件
    create_and_save_metadata(chunks_docs,FAISS_DB_PATH,METADATA_FILE_NAME)

    print("\n ✅ 进程成功完成！ ")
    print(f"📌 FAISS 索引和元数据存储在：{os.path.abspath(FAISS_DB_PATH)}")

    print("\n --- Rerank 功能示例 ---")
    loaded_model = get_embeddings_model(EMBEDDING_MODEL)
    loaded_faiss_db = FAISS.load_local(FAISS_DB_PATH, loaded_model, allow_dangerous_deserialization=True)
    print(f"🔎 已从 {FAISS_DB_PATH} 加载 FAISS 数据库。")

    while True:
        user_query = input("\n💬 请输入你的法律问题（输入 q 退出）：\n>>> ").strip()
        if user_query.lower() in ["q", "quit", "exit"]:
            print("👋 已退出。")
            break
        if not user_query:
            print("⚠️ 输入不能为空。")
            continue

        retriever = loaded_faiss_db.as_retriever(search_kwargs={"k": 8})
        initial_retrieved_docs = retriever.invoke(user_query)
        # print(f"\n--- 🔍 初始检索到 {len(initial_retrieved_docs)} 篇文档 ---")
        # for i, doc in enumerate(initial_retrieved_docs):
        #     print(f" 初始 Top {i + 1}: {doc.page_content[:100]}...")

        top_reranked_docs = rerank_documents_siliconflow(user_query, initial_retrieved_docs, top_n=5)

        print("\n🚀 正在调用大模型 LLM 生成最终答案...")
        final_answer = generate_llm_response(user_query, top_reranked_docs)
        print(f"\n📢 回答：\n{final_answer}")



    # 调用 Reranker 对初步检索到的文档进行重排
    # top_reranked_docs = rerank_documents_siliconflow(user_query, initial_retrieved_docs, top_n=3)
    # if top_reranked_docs:
    #     print("\n--- 📚 重排后的文档（用于最终答案生成）---")
    #     for i, doc in enumerate(top_reranked_docs):
    #         print(f"最终 Top {i + 1}: {doc.page_content[:200]}...")
    #
    #     print("\n🚀 正在调用大模型 LLM 生成最终答案...")
    #     final_answer = generate_llm_response(user_query, top_reranked_docs)
    #     print(f"\n💬 问题：{user_query}")
    #     print("📢 LLM 生成的总结回答：")
    #     print("-" * 60)
    #     print(final_answer)
    #     print("-" * 60)
    #
    # else:
    #     print("\n未能成功重排文档，请检查 API 密钥和网络连接。")