import warnings
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM, pipeline
from typing import List, Optional
import torch
import os
from dotenv import load_dotenv
import requests
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging
from contextlib import asynccontextmanager
import sys


load_dotenv()
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(LLM):
    max_token: int = 3096
    temperature: float = 0.7
    top_p : float = 0.9             # 这里不用top-k，而是概率，即知识库中，待选的chunks，必须与query的相似度之和必须小于等于0.9
    base_url: str = "https://api.siliconflow.cn/v1/" 
    api_key: str = SILICONFLOW_API_KEY  
    history: list = [] 
    model: object = None    # 模型对象
    
    def __init__(self):
        super().__init__()
        self._initialize_model()
        
        
    def _initialize_model(self):
        try:
            self.model = ChatOpenAI(
                model="Qwen/QwQ-32B",
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_token
            )
            logger.info("模型初始化成功！")
        except Exception as e:
            logger.error("模型初始化失败:%s", e)
            self.model = None
            
    def get_history(self):
        return self.history

    def set_history(self, history):
        self.history = history
        
    @property
    def _llm_type(self) -> str:
        return "BaseModel"
    
    def _call(self,prompt: str, stop: Optional[List[str]] = None) -> str:
        
        history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in self.history])
        # 初始化消息列表，添加系统消息
        messages = [
            ("system", f'''
             你是一位专业素养极高的律师，能够精准把握法律条文，为客户提供精炼且准确的法律咨询服务。
             请密切关注历史对话记录，结合当前上下文进行回答，确保回答的连贯性和准确性。

            历史对话记录：
            {history_text}
             ''')
        ]
        # # 添加历史对话记录到消息列表
        for entry in self.history:
            role = "human" if entry["role"] == "user" else "assistant"
            messages.append((role, entry["content"]))
            
        # 添加当前用户的提问
        messages.append(("human", prompt))
        # 创建 ChatPromptTemplate
        # chainA_template = ChatPromptTemplate.from_messages(messages)
        # # 创建一个字符串输出解析器
        # output_parser = StrOutputParser()
        
        try:
            # chainA = chainA_template | self.model | output_parser
            response = self.model.invoke(messages)
            generated_text = response.content if hasattr(response, "content") else str(response)

            
            # 如果有停止词，则应用停止词
            if stop:
                generated_text = enforce_stop_tokens(generated_text, stop)

            # 更新对话历史
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": generated_text})
            
            # 当历史对话超过5条，删除最早的记录
            if len(self.history) > 5:
                self.history.pop(0)

            return generated_text
        except Exception as e:
            logger.error(f"调用 chainA 出错: {e}")
            return f"调用出错: {str(e)}"
        
        
class Qwen_32B(BaseModel):
    @property
    def _llm_type(self) -> str:
        return "Qwen_32B"

    def _initialize_model(self):
        try:
            self.model = ChatOpenAI(
                model="Qwen/QwQ-32B",
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_token
            )
            logger.info("Qwen_32B 模型初始化成功！")
        except Exception as e:
            logger.error("Qwen_32B 模型初始化失败:%s", e)
            self.model = None


class DeepSeek_R1(BaseModel):
    @property
    def _llm_type(self) -> str:
        return "DeepSeek-R1"

    def _initialize_model(self):
        try:
            self.model = ChatOpenAI(
                model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_token
            )
            logger.info("DeepSeek-R1 模型初始化成功！")
        except Exception as e:
            logger.error("DeepSeek-R1 模型初始化失败:%s", e)
            self.model = None


class GLM_V4(BaseModel):
    @property
    def _llm_type(self) -> str:
        return "GLM-V4"

    def _initialize_model(self):
        try:
            self.model = ChatOpenAI(
                model="THUDM/GLM-4.1V-9B-Thinking",
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_token
            )
            logger.info("GLM_V4 模型初始化成功！")
        except Exception as e:
            logger.error("GLM_V4 模型初始化失败:%s", e)
            self.model = None

if __name__ == '__main__':
    # =========== DeepSeek使用示例 ===========
    # llm = DeepSeek_R1()
    
    # =========== Qwen_32B使用示例 ===========
    # llm = Qwen_32B()
    
    # =========== GLM_V4使用示例 ===========
    llm=GLM_V4()
    
    
    print("Q1:1+1=?--",llm.invoke("1+1等于几？"))
    print("Q2:上一个问题是什么--",llm.invoke("我的上一个问题是什么？"))