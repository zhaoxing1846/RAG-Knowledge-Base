"""
向量存储 (FAISS) - 替代 ChromaDB
"""
import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import CACHE_DIR
from embeddings import EmbeddingClient


class VectorStore:
    """FAISS 向量存储封装"""
    
    def __init__(self, persist_dir: str = None, dimension: int = 4096):
        self.persist_dir = persist_dir or os.path.join(CACHE_DIR, "faiss")
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.dimension = dimension
        self.embedder = EmbeddingClient()
        
        # FAISS 索引 (使用 Inner Product / 余弦相似度)
        self.index = faiss.IndexFlatIP(dimension)
        
        # 文档存储
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self.ids: List[str] = []
        
        # 加载已有数据
        self._load()
    
    def _load(self):
        """从磁盘加载索引和文档"""
        index_path = os.path.join(self.persist_dir, "index.faiss")
        data_path = os.path.join(self.persist_dir, "data.pkl")
        
        if os.path.exists(index_path) and os.path.exists(data_path):
            self.index = faiss.read_index(index_path)
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                self.metadatas = data.get("metadatas", [])
                self.ids = data.get("ids", [])
            print(f"Loaded {len(self.documents)} documents from disk")
    
    def _save(self):
        """保存索引和文档到磁盘"""
        faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
        with open(os.path.join(self.persist_dir, "data.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas,
                "ids": self.ids
            }, f)
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[dict] = None,
        ids: List[str] = None
    ) -> int:
        """添加文档"""
        if not documents:
            return 0
        
        if ids is None:
            ids = [f"doc_{len(self.ids) + i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # 获取向量
        embeddings = self.embedder.embed(documents)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # 归一化（用于余弦相似度）
        faiss.normalize_L2(embeddings_np)
        
        # 添加到索引
        self.index.add(embeddings_np)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        # 保存
        self._save()
        
        return len(documents)
    
    def add_pdf_chunks(self, pdf_result: Dict) -> int:
        """添加 PDF chunks"""
        chunks = pdf_result.get("chunks", [])
        if not chunks:
            return 0
        
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{pdf_result['filename']}_chunk_{c['chunk_id']}"
            for c in chunks
        ]
        
        return self.add_documents(documents, metadatas, ids)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict = None
    ) -> List[Dict]:
        """搜索相似文档"""
        if self.index.ntotal == 0:
            return []
        
        # 获取查询向量
        query_embedding = self.embedder.embed_single(query)
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # 搜索
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_np, k)
        
        hits = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            
            doc_metadata = self.metadatas[idx]
            
            # 元数据过滤
            if where:
                match = all(
                    doc_metadata.get(k) == v
                    for k, v in where.items()
                )
                if not match:
                    continue
            
            hits.append({
                "id": self.ids[idx],
                "text": self.documents[idx],
                "metadata": doc_metadata,
                "score": float(scores[0][i]),  # 余弦相似度
                "distance": 1 - float(scores[0][i])  # 转换为距离
            })
        
        return hits
    
    def delete_by_source(self, source: str):
        """删除指定来源的文档（需要重建索引）"""
        # FAISS 不支持删除，需要重建
        keep_indices = [
            i for i, m in enumerate(self.metadatas)
            if m.get("source") != source
        ]
        
        if not keep_indices:
            self.clear()
            return
        
        # 提取保留的文档
        new_docs = [self.documents[i] for i in keep_indices]
        new_metas = [self.metadatas[i] for i in keep_indices]
        new_ids = [self.ids[i] for i in keep_indices]
        
        # 重建索引
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # 重新添加
        self.add_documents(new_docs, new_metas, new_ids)
    
    def clear(self):
        """清空索引"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []
        self._save()
    
    def count(self) -> int:
        """文档数量"""
        return self.index.ntotal
    
    def save(self):
        """公开保存方法"""
        self._save()

    def list_documents(self) -> List[str]:
        """获取所有文档文件名（去重）"""
        sources = set()
        for meta in self.metadatas:
            src = meta.get("source", "")
            if src:
                sources.add(src)
        return sorted(list(sources))

    def get_stats(self) -> Dict:
        """统计信息"""
        return {
            "total_documents": self.count(),
            "dimension": self.dimension,
            "persist_directory": self.persist_dir
        }


if __name__ == "__main__":
    store = VectorStore()
    print(f"Stats: {store.get_stats()}")
