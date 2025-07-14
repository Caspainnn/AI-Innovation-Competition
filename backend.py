from fastapi import FastAPI, Request
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 引入你已有的核心模块函数
from RAG import (
    get_embeddings_model,rerank_documents_siliconflow, generate_llm_response
)

app = FastAPI()

# 加载向量数据库一次，避免每次请求重新加载
embedding_model = get_embeddings_model("BAAI/bge-small-zh-v1.5")
faiss_db = FAISS.load_local("./faiss_index", embedding_model,allow_dangerous_deserialization=True)

class QueryRequest(BaseModel):
    query: str

class Reference(BaseModel):
    source: str
    preview: str
    score: float

class AnswerResponse(BaseModel):
    answer: str
    references: List[Reference]

@app.get("/")
def read_root():
    return {"message": "🎯 法务智能问答系统"}


@app.post("/query", response_model=AnswerResponse)
async def rag_query(request: QueryRequest):
    query = request.query.strip()

    retriever = faiss_db.as_retriever(search_kwargs={"k": 4})
    retrieved_docs = retriever.invoke(query)
    reranked_docs = rerank_documents_siliconflow(query, retrieved_docs, top_n=5)

    answer = generate_llm_response(query, reranked_docs)

    # 🔍 封装所有参考文档（可选：retrieved_docs 或 reranked_docs）
    references = []
    for doc , score in reranked_docs:  # 或 reranked_docs，根据你想展示哪些
        references.append({
            "source": doc.metadata.get("source", "未知"),
            "preview": doc.page_content[:150] + "...",
            "score": score
        })

    print(f"参考文档返回数量：{len(references)}")
    print("📦 返回内容预览：")
    print({"answer": answer, "references": references})

    return {
        "answer": answer,
        "references": references
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
