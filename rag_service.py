"""
RAG 服务 - FastAPI

启动方式:
    uvicorn rag_service:app --host 0.0.0.0 --port 8000

接口:
    GET  /              - Web 聊天界面
    POST /api/chat      - 多轮对话
    POST /api/chat/stream - 流式对话（避免超时）
    POST /search        - 搜索文档
    POST /ask           - 单次问答
    GET  /status        - 服务状态
    POST /embed         - 处理并索引 PDF
    POST /clear         - 清空索引
"""
import os
import asyncio
import time
import logging
import json
import queue
import threading
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/rag_service.log')
    ]
)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import PDF_DIR, TOP_K_SIMILAR
from pdf_processor import process_pdf, process_pdf_directory
from vector_store import VectorStore
from embeddings import RerankerClient, LLMClient
from user_manager import register_user, login_user, get_current_user
from conversation_manager import (create_conversation, get_conversations,
                                   delete_conversation, get_messages, add_message)

# ==================== FastAPI App ====================

app = FastAPI(
    title="RAG Service",
    description="基于 Qwen 的本地知识库 RAG 服务",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
vector_store: Optional[VectorStore] = None
reranker: Optional[RerankerClient] = None
llm: Optional[LLMClient] = None

# 静态文件目录
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.on_event("startup")
async def startup():
    global vector_store, reranker, llm
    vector_store = VectorStore()
    reranker = RerankerClient()
    llm = LLMClient()
    print(f"RAG Service started. Documents in store: {vector_store.count()}")


# ==================== 请求模型 ====================

class EmbedRequest(BaseModel):
    pdf_path: Optional[str] = None   # 单个 PDF 路径
    pdf_dir: Optional[str] = None    # PDF 目录
    clear_existing: bool = False     # 是否清空已有索引


class SearchRequest(BaseModel):
    query: str
    top_k: int = TOP_K_SIMILAR
    use_reranker: bool = True


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    use_reranker: bool = True
    system_prompt: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []  # [{"role": "user/assistant", "content": "..."}]
    top_k: int = 5


class SourceRequest(BaseModel):
    source: str


# ==================== API 路由 ====================

@app.get("/")
async def root():
    """返回聊天界面"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/status")
async def get_status():
    """获取服务状态"""
    return {
        "status": "running",
        "vector_store": vector_store.get_stats() if vector_store else None
    }


@app.post("/embed")
async def embed_documents(req: EmbedRequest):
    """
    处理 PDF 并添加到向量库
    
    - 如果指定 pdf_path，处理单个文件
    - 如果指定 pdf_dir，处理整个目录
    - 都不指定则使用默认 PDF_DIR
    """
    if req.clear_existing:
        vector_store.clear()
        print("Cleared existing index")
    
    results = []
    
    if req.pdf_path:
        # 单个文件
        result = process_pdf(req.pdf_path)
        count = vector_store.add_pdf_chunks(result)
        results.append({"file": result["filename"], "chunks_added": count})
    
    elif req.pdf_dir:
        # 指定目录
        pdf_results = process_pdf_directory(req.pdf_dir)
        for r in pdf_results:
            count = vector_store.add_pdf_chunks(r)
            results.append({"file": r["filename"], "chunks_added": count})
    
    else:
        # 默认目录
        pdf_results = process_pdf_directory(PDF_DIR)
        for r in pdf_results:
            count = vector_store.add_pdf_chunks(r)
            results.append({"file": r["filename"], "chunks_added": count})
    
    return {
        "indexed": results,
        "total_documents": vector_store.count()
    }


@app.post("/search")
async def search_documents(req: SearchRequest):
    """搜索相似文档"""
    hits = vector_store.search(req.query, top_k=req.top_k * 2)  # 多取一些用于 rerank
    
    if req.use_reranker and hits:
        # Rerank
        documents = [h["text"] for h in hits]
        rerank_results = reranker.rerank(req.query, documents, top_k=req.top_k)
        
        # 按 rerank 结果重排
        reranked_hits = []
        for r in rerank_results:
            idx = r["index"]
            hit = hits[idx].copy()
            hit["rerank_score"] = r["relevance_score"]
            reranked_hits.append(hit)
        
        return {"query": req.query, "results": reranked_hits}
    
    # 不使用 reranker，按距离排序
    hits_sorted = sorted(hits, key=lambda x: x["distance"])
    return {"query": req.query, "results": hits_sorted[:req.top_k]}


@app.post("/ask")
async def ask_question(req: AskRequest):
    """RAG 问答"""
    # 搜索相关文档
    hits = vector_store.search(req.question, top_k=req.top_k * 2)
    
    if not hits:
        return {
            "question": req.question,
            "answer": "抱歉，知识库中没有找到相关内容。",
            "sources": []
        }
    
    # Rerank
    if req.use_reranker:
        documents = [h["text"] for h in hits]
        rerank_results = reranker.rerank(req.question, documents, top_k=req.top_k)
        top_hits = [hits[r["index"]] for r in rerank_results]
    else:
        top_hits = sorted(hits, key=lambda x: x["distance"])[:req.top_k]
    
    # 提取上下文
    contexts = [h["text"] for h in top_hits]
    sources = list(set(h["metadata"].get("source", "unknown") for h in top_hits))
    
    # 生成回答
    answer = llm.generate_answer(
        req.question,
        contexts,
        system_prompt=req.system_prompt
    )
    
    return {
        "question": req.question,
        "answer": answer,
        "sources": sources,
        "context_count": len(contexts)
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """多轮对话 RAG - 带超时保护和详细日志"""
    start_time = time.time()
    logger.info(f"[CHAT] 新请求：问题长度={len(req.question)}, top_k={req.top_k}")
    
    try:
        # 搜索相关文档
        t0 = time.time()
        hits = vector_store.search(req.question, top_k=req.top_k * 2)
        logger.info(f"[CHAT] 向量搜索完成：{time.time()-t0:.2f}s, 命中={len(hits)}")
        
        if not hits:
            logger.warning("[CHAT] 无相关文档")
            return {
                "answer": "抱歉，知识库中没有找到与问题相关的内容。",
                "sources": []
            }
        
        # Rerank
        t1 = time.time()
        documents = [h["text"] for h in hits]
        rerank_results = reranker.rerank(req.question, documents, top_k=req.top_k)
        top_hits = [hits[r["index"]] for r in rerank_results]
        logger.info(f"[CHAT] Rerank 完成：{time.time()-t1:.2f}s")
        
        # 提取上下文（含来源去重）
        contexts = [h["text"] for h in top_hits]
        seen_files = set()
        unique_sources = []
        context_items = []
        for i, (ctx, hit) in enumerate(zip(contexts, top_hits)):
            src = hit["metadata"].get("source", "未知文件")
            ref_key = f"[{src}]"
            if src not in seen_files:
                seen_files.add(src)
                unique_sources.append(src)
            context_items.append(f"{ref_key}\n{ctx}")
        context_text = "\n\n".join(context_items)

        # 构建系统提示
        system_prompt = """你是一个学术研究助手，专门回答关于智能嗅觉、高通量筛选、纳米材料等领域的问题。
请基于提供的参考文献内容回答问题。如果参考文献中没有相关信息，请诚实说明。
回答要清晰、专业。回答中引用来源时，使用文件名格式如 [文件名.pdf] 来标注。
如果是追问，请结合之前的对话上下文来理解用户意图。"""
        
        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史对话
        for msg in req.history or []:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加当前问题和上下文
        messages.append({
            "role": "user", 
            "content": f"参考文献：\n{context_text}\n\n问题：{req.question}"
        })
        
        # 生成回答 - 带超时保护
        t2 = time.time()
        logger.info(f"[CHAT] 开始调用 LLM，消息数={len(messages)}")
        try:
            answer = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: llm.chat(messages, temperature=0.6)
                ),
                timeout=300.0  # 300 秒超时（5分钟）
            )
            logger.info(f"[CHAT] LLM 响应完成：{time.time()-t2:.2f}s, 回答长度={len(answer)}")
        except asyncio.TimeoutError:
            logger.error("[CHAT] LLM 响应超时 (120s)")
            return JSONResponse(
                status_code=504,
                content={
                    "answer": "抱歉，回答生成超时。问题比较复杂，请尝试简化问题或分步提问。",
                    "sources": unique_sources,
                    "timeout": True
                },
                headers={"X-Elapsed-Time": f"{time.time()-start_time:.1f}s"}
            )

        answer = re.sub(r'</?br\s*/?>', '  \n', answer)
        source_lines = [f"- {s}" for s in unique_sources]
        answer += f"\n\n📄 **来源文件**\n" + "\n".join(source_lines)

        total_time = time.time() - start_time
        logger.info(f"[CHAT] 请求完成：总耗时={total_time:.2f}s")

        return {
            "answer": answer,
            "sources": unique_sources,
            "elapsed_time": f"{total_time:.2f}s"
        }

    except Exception as e:
        import traceback
        logger.error(f"[CHAT] 错误：{str(e)}\n{traceback.format_exc()}")
        return {
            "answer": f"抱歉，处理请求时出错：{str(e)[:100]}",
            "sources": [],
            "error": True
        }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式对话 - 逐 token SSE 流式输出 + 引用标注"""
    start_time = time.time()
    logger.info(f"[CHAT-STREAM] 新请求：问题长度={len(req.question)}")

    async def generate():
        try:
            hits = vector_store.search(req.question, top_k=req.top_k * 2)

            if not hits:
                yield f"data: {json.dumps({'type': 'error', 'content': '抱歉，知识库中没有找到与问题相关的内容。'})}\n\n"
                return

            documents = [h["text"] for h in hits]
            rerank_results = reranker.rerank(req.question, documents, top_k=req.top_k)
            top_hits = [hits[r["index"]] for r in rerank_results]

            contexts = [h["text"] for h in top_hits]
            seen_files = set()
            unique_sources = []
            context_items = []
            for i, (ctx, hit) in enumerate(zip(contexts, top_hits)):
                src = hit["metadata"].get("source", "未知文件")
                ref_key = f"[{src}]"
                if src not in seen_files:
                    seen_files.add(src)
                    unique_sources.append(src)
                context_items.append(f"{ref_key}\n{ctx}")
            context_text = "\n\n".join(context_items)

            yield f"data: {json.dumps({'type': 'sources', 'content': unique_sources})}\n\n"

            system_prompt = """你是一个学术研究助手，专门回答关于智能嗅觉、高通量筛选、纳米材料等领域的问题。
请基于提供的参考文献内容回答问题。如果参考文献中没有相关信息，请诚实说明。
回答要清晰、专业。回答中引用来源时，使用文件名格式如 [文件名.pdf] 来标注。
如果是追问，请结合之前的对话上下文来理解用户意图。"""

            messages = [{"role": "system", "content": system_prompt}]
            for msg in req.history or []:
                if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({
                "role": "user",
                "content": f"背景资料：\n{context_text}\n\n问题：{req.question}"
            })

            yield f"data: {json.dumps({'type': 'status', 'content': '正在生成回答...'})}\n\n"

            # 逐 token 流式输出（线程队列）
            token_queue = queue.Queue()

            def _stream_worker():
                for token in llm.chat_stream(messages, temperature=0.6):
                    token_queue.put(token)
                token_queue.put(None)

            threading.Thread(target=_stream_worker, daemon=True).start()

            full_answer_parts = []
            while True:
                token = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: token_queue.get()
                )
                if token is None:
                    break
                full_answer_parts.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            answer = "".join(full_answer_parts)
            answer = re.sub(r'</?br\s*/?>', '  \n', answer)

            # 后端追加来源文件列表到末尾
            source_lines = [f"- {s}" for s in unique_sources]
            answer += f"\n\n📄 **来源文件**\n" + "\n".join(source_lines)

            yield f"data: {json.dumps({'type': 'answer', 'content': answer})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'elapsed': f'{time.time()-start_time:.1f}s'})}\n\n"

        except Exception as e:
            logger.error(f"[CHAT-STREAM] 错误：{str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'处理出错：{str(e)[:50]}'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/delete")
async def delete_by_source(req: SourceRequest):
    """删除指定来源的所有文档"""
    vector_store.delete_by_source(req.source)
    return {"deleted": req.source, "remaining": vector_store.count()}


@app.post("/clear")
async def clear_index():
    """清空索引"""
    vector_store.clear()
    return {"status": "cleared", "documents": 0}


# ==================== 用户认证 API ====================

class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/api/register")
async def api_register(req: AuthRequest):
    return register_user(req.username, req.password)


@app.post("/api/login")
async def api_login(req: AuthRequest):
    return login_user(req.username, req.password)


# ==================== 上传 PDF ====================

import tempfile


@app.post("/api/upload")
async def api_upload_pdf(file: UploadFile = File(...)):
    """上传 PDF 并自动索引到知识库"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 保存到 PDF 目录
    os.makedirs(PDF_DIR, exist_ok=True)
    save_path = os.path.join(PDF_DIR, file.filename)

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"[UPLOAD] 已保存：{save_path} ({len(content)} bytes)")

    # 处理并索引
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: process_pdf(save_path))
        count = vector_store.add_pdf_chunks(result)
        vector_store.save()
        logger.info(f"[UPLOAD] 已索引：{file.filename} → {count} 个片段")
        return {
            "success": True,
            "filename": file.filename,
            "size": len(content),
            "chunks_added": count,
            "total_documents": vector_store.count()
        }
    except Exception as e:
        logger.error(f"[UPLOAD] 索引失败：{e}")
        raise HTTPException(status_code=500, detail=f"PDF 处理失败：{str(e)[:100]}")


@app.get("/api/documents")
async def api_get_documents():
    """获取知识库中所有文档列表"""
    docs = vector_store.list_documents()
    return {"documents": docs, "total": len(docs)}


# ==================== 管理员 API ====================

def _require_admin(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录")
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


class RoleUpdateRequest(BaseModel):
    user_id: int
    role: str


@app.get("/api/admin/users")
async def api_admin_get_users(admin: dict = Depends(_require_admin)):
    from user_manager import get_all_users
    users = get_all_users()
    # 隐藏密码哈希等敏感信息
    return {"users": users, "total": len(users)}


@app.put("/api/admin/users/role")
async def api_admin_update_role(req: RoleUpdateRequest, admin: dict = Depends(_require_admin)):
    from user_manager import update_user_role
    return update_user_role(req.user_id, req.role)


@app.delete("/api/admin/users/{user_id}")
async def api_admin_delete_user(user_id: int, admin: dict = Depends(_require_admin)):
    from user_manager import delete_user
    return delete_user(user_id)


# ==================== 对话管理 API ====================

class ConvCreateRequest(BaseModel):
    title: str = "新对话"

class MessageRequest(BaseModel):
    role: str
    content: str

class ConvUpdateRequest(BaseModel):
    title: str


def _require_user(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return user


@app.get("/api/conversations")
async def api_get_conversations(user: dict = Depends(_require_user)):
    return get_conversations(user["user_id"])


@app.post("/api/conversations")
async def api_create_conversation(
    request: ConvCreateRequest,
    user: dict = Depends(_require_user)
):
    return create_conversation(user["user_id"], request.title)


@app.delete("/api/conversations/{conv_id}")
async def api_delete_conversation(conv_id: str, user: dict = Depends(_require_user)):
    success = delete_conversation(conv_id, user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="对话不存在或无权限")
    return {"success": True}


@app.get("/api/conversations/{conv_id}/messages")
async def api_get_messages(conv_id: str, user: dict = Depends(_require_user)):
    msgs = get_messages(conv_id, user["user_id"])
    if msgs is None:
        raise HTTPException(status_code=404, detail="对话不存在或无权限")
    return msgs


@app.post("/api/conversations/{conv_id}/messages")
async def api_add_message(
    conv_id: str,
    request: MessageRequest,
    user: dict = Depends(_require_user)
):
    if not request.content:
        raise HTTPException(status_code=400, detail="消息内容不能为空")
    msg = add_message(conv_id, user["user_id"], request.role, request.content)
    if msg is None:
        raise HTTPException(status_code=404, detail="对话不存在或无权限")
    return msg


@app.patch("/api/conversations/{conv_id}")
async def api_update_conversation(
    conv_id: str,
    request: ConvUpdateRequest,
    user: dict = Depends(_require_user)
):
    from conversation_manager import _get_db
    conn = _get_db()
    row = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, user["user_id"])
    ).fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="对话不存在或无权限")
    conn.execute(
        "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
        (request.title, conv_id)
    )
    conn.commit()
    conn.close()
    return {"success": True}


# ==================== 运行 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
