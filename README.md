# RAG 知识库

基于 Qwen 大模型的本地学术论文知识库问答系统。

## 功能

- PDF 文档自动解析、向量化索引
- 语义搜索 + Rerank 重排序
- 基于 RAG 的多轮对话问答
- 流式 SSE 响应（避免超时）
- 用户注册/登录系统
- 多对话管理（新建、切换、删除）
- Web SPA 界面（纯原生 JS，无框架依赖）

## 快速开始

```bash
# 1. 配置 API key
export RAG_API_KEY="your-api-key"

# 2. 安装依赖
pip install fastapi uvicorn pydantic

# 3. 索引 PDF 文档
python3 -c "
from pdf_processor import process_pdf_directory
from vector_store import VectorStore
results = process_pdf_directory('PDF/')
store = VectorStore()
for r in results:
    store.add_pdf_chunks(r)
print(f'Indexed {store.count()} chunks')
"

# 4. 启动服务
python3 -c "
import rag_service
import uvicorn
uvicorn.run(rag_service.app, host='0.0.0.0', port=8080)
"
```

## 项目结构

```
pdf_pipeline/
├── rag_service.py            # FastAPI 主服务
├── user_manager.py           # 用户认证（JWT）
├── conversation_manager.py   # 对话管理（SQLite）
├── embeddings.py             # 向量嵌入 + Reranker + LLM 客户端
├── pdf_processor.py          # PDF 解析
├── vector_store.py           # FAISS 向量存储
├── config.py                 # 配置（API key 从环境变量读取）
├── static/index.html         # Web 前端 SPA
└── cache/                    # 索引缓存
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/register` | 用户注册 |
| POST | `/api/login` | 用户登录 |
| GET | `/api/conversations` | 对话列表 |
| POST | `/api/conversations` | 创建对话 |
| PATCH | `/api/conversations/{id}` | 更新对话标题 |
| DELETE | `/api/conversations/{id}` | 删除对话 |
| GET | `/api/conversations/{id}/messages` | 获取消息 |
| POST | `/api/conversations/{id}/messages` | 添加消息 |
| POST | `/api/chat/stream` | 流式问答 |
| POST | `/search` | 语义搜索 |
| POST | `/embed` | 索引 PDF |
| GET | `/status` | 服务状态 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_API_KEY` | API key | 必须设置 |
| `RAG_API_BASE` | API 地址 | `https://uni-api.cstcloud.cn/v1` |
| `RAG_WORKSPACE` | 工作目录 | 项目上级目录 |

## 数据格式

MCU 数据格式（逗号分隔文本）：`25.3,60,1013`

每行一条数据，换行分隔。