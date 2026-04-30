#!/bin/sh
# RAG 服务启动脚本 - 设置正确的 API key
export RAG_API_KEY="353ae8ca32aeff85742426f9d3de839c1f94b4ae8008a54fbd84e22ed1f1d7c3"
export RAG_API_BASE="https://uni-api.cstcloud.cn/v1"
cd /root/.openclaw/workspace/pdf_pipeline
exec python3 -m uvicorn rag_service:app --host 127.0.0.1 --port 8080