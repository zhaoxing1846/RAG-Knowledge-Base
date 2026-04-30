# RAG 知识库

基于 Qwen 大模型的本地学术论文知识库问答系统，支持流式输出、用户管理和 PDF 上传。

## 功能

- 📄 **PDF 自动索引** — 解析、分割、向量化存储（FAISS）
- 🔍 **语义检索 + Rerank** — 精准重排序优化检索质量
- ⚡ **流式 SSE 响应** — 逐块推送，无需等待完整回答
- 👤 **用户认证系统** — JWT 注册/登录
- 🔐 **超级管理员面板** — 用户管理、文档管理
- 📤 **Web PDF 上传** — 管理员直接上传新文档并自动索引
- 💬 **多轮对话** — 保存对话历史
- 🖥️ **Web SPA 界面** — 纯原生 JS，无框架依赖

## 快速开始

```bash
# 1. 安装依赖
pip install fastapi uvicorn pydantic httpx faiss-cpu python-pdfminer pillow

# 2. 配置 API key
export RAG_API_KEY="your-api-key"
export RAG_API_BASE="https://uni-api.cstcloud.cn/v1"

# 3. 放入 PDF 文件
# 把 PDF 文件放到 PDF/ 目录

# 4. 启动服务
systemctl start rag   # systemd（推荐）
# 或
sh start_rag.sh       # 手动
# 或
python3 -m uvicorn rag_service:app --host 0.0.0.0 --port 8080
```

访问 `http://localhost:8080`

## systemd 服务（生产部署）

```bash
sudo cp rag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rag
sudo systemctl start rag
```

## 配置

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `RAG_API_KEY` | API key（必需） | - |
| `RAG_API_BASE` | API 地址 | `https://uni-api.cstcloud.cn/v1` |
| `RAG_WORKSPACE` | 工作目录 | 项目上级目录 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | Web 界面 |
| GET | `/status` | 服务状态 |
| POST | `/api/register` | 用户注册 |
| POST | `/api/login` | 用户登录 |
| POST | `/api/chat/stream` | ⚡ 流式问答 |
| POST | `/api/upload` | 📤 上传 PDF |
| GET | `/api/documents` | 文档列表 |
| GET | `/api/conversations` | 对话列表 |
| POST | `/api/conversations` | 创建对话 |
| DELETE | `/api/conversations/{id}` | 删除对话 |
| GET | `/api/conversations/{id}/messages` | 获取消息 |
| GET | `/api/admin/users` | 👑 用户列表（管理员） |
| PUT | `/api/admin/users/role` | 👑 修改用户角色（管理员） |
| DELETE | `/api/admin/users/{id}` | 👑 删除用户（管理员） |

## 项目结构

```
pdf_pipeline/
├── rag_service.py            # FastAPI 主服务
├── user_manager.py           # 用户认证（JWT + 管理员）
├── conversation_manager.py   # 对话管理（SQLite）
├── embeddings.py             # 向量嵌入 + Reranker + LLM 客户端
├── pdf_processor.py          # PDF 解析
├── vector_store.py           # FAISS 向量存储
├── config.py                 # 配置
├── start_rag.sh              # 启动脚本
├── static/index.html         # Web 前端 SPA
├── cache/                    # FAISS 索引缓存
└── PDF/                      # PDF 文档目录
```

## 默认管理员

- 用户名：`admin`
- 密码：`admin`（生产环境请尽快修改）