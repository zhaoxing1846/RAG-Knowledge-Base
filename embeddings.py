"""
Embedding 和 Reranker 调用
"""
import os
import json
import httpx
from typing import List, Optional
import hashlib


# 从 config 导入
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    API_BASE, API_KEY,
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL
)


class EmbeddingClient:
    """OpenAI 兼容的 Embedding API 客户端"""

    def __init__(
        self,
        api_base: str = API_BASE,
        api_key: str = API_KEY,
        model: str = EMBEDDING_MODEL
    ):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """获取文本向量"""
        url = f"{self.api_base}/embeddings"

        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "input": texts,
                "encoding_format": "float"
            },
            timeout=60.0
        )

        if response.status_code != 200:
            raise Exception(f"Embedding API error: {response.status_code} - {response.text}")

        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    def embed_single(self, text: str) -> List[float]:
        """获取单个文本的向量"""
        return self.embed([text])[0]


class RerankerClient:
    """Reranker API 客户端"""

    def __init__(
        self,
        api_base: str = API_BASE,
        api_key: str = API_KEY,
        model: str = RERANKER_MODEL
    ):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[dict]:
        """
        对文档重排序

        返回: [{"index": int, "relevance_score": float}, ...]
        """
        url = f"{self.api_base}/rerank"

        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k
            },
            timeout=60.0
        )

        if response.status_code != 200:
            raise Exception(f"Reranker API error: {response.status_code} - {response.text}")

        data = response.json()
        return data.get("results", [])


class LLMClient:
    """LLM API 客户端"""

    def __init__(
        self,
        api_base: str = API_BASE,
        api_key: str = API_KEY,
        model: str = LLM_MODEL
    ):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 300  # 增加到 5 分钟
    ) -> str:
        """对话补全（非流式，返回完整文本）"""
        url = f"{self.api_base}/chat/completions"

        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            },
            timeout=timeout
        )

        if response.status_code != 200:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 300
    ) -> str:
        """流式对话补全，逐 chunk 返回文本内容"""
        url = f"{self.api_base}/chat/completions"

        with httpx.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            },
            timeout=timeout
        ) as response:
            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code}")
            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if data_str:
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices")
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except (json.JSONDecodeError, IndexError, ValueError):
                            pass

    def generate_answer(
        self,
        question: str,
        contexts: List[str],
        system_prompt: str = None
    ) -> str:
        """基于上下文生成回答"""
        if system_prompt is None:
            system_prompt = """你是一个学术研究助手。请基于提供的参考文献内容回答问题。
如果参考文献中没有相关信息，请诚实说明。
引用时请注明来源（如 [文档1]）。"""

        context_text = "\n\n".join([
            f"[文档{i+1}]\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"参考文献：\n{context_text}\n\n问题：{question}"}
        ]

        return self.chat(messages, temperature=0.5)


if __name__ == "__main__":
    # 测试
    emb = EmbeddingClient()
    test_vec = emb.embed_single("测试文本")
    print(f"Embedding dimension: {len(test_vec)}")
    print(f"First 5 values: {test_vec[:5]}")
