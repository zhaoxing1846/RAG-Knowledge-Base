#!/bin/bash
# RAG 服务监控脚本

LOGFILE="/tmp/rag_monitor.log"
echo "=== RAG 监控开始 $(date) ===" >> $LOGFILE

# 检查服务状态
echo "服务状态:" >> $LOGFILE
sudo systemctl status rag --no-pager >> $LOGFILE 2>&1

# 检查端口
echo -e "\n端口占用:" >> $LOGFILE
sudo ss -tlnp | grep 8080 >> $LOGFILE 2>&1

# 最近日志
echo -e "\nRAG 最近日志:" >> $LOGFILE
sudo journalctl -u rag --no-pager -n 20 >> $LOGFILE 2>&1

# Nginx 错误
echo -e "\nNginx RAG 错误:" >> $LOGFILE
tail -20 /var/log/nginx/rag-error.log >> $LOGFILE 2>&1

# 进程内存
echo -e "\nRAG 进程:" >> $LOGFILE
ps aux | grep -E "8080|rag_service" | grep -v grep >> $LOGFILE 2>&1

echo "" >> $LOGFILE
