"""
对话管理模块 - 对话和消息的 CRUD
使用标准库实现，无需额外依赖
"""
import os
import sqlite3
import uuid
from typing import List, Optional

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "rag_conversations.db")


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL DEFAULT '新对话',
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at DESC);
    """)
    conn.commit()
    conn.close()


def create_conversation(user_id: int, title: str = "新对话") -> dict:
    conv_id = uuid.uuid4().hex
    conn = _get_db()
    conn.execute(
        "INSERT INTO conversations (id, user_id, title) VALUES (?, ?, ?)",
        (conv_id, user_id, title)
    )
    conn.commit()
    row = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
        (conv_id,)
    ).fetchone()
    conn.close()
    return dict(row)


def get_conversations(user_id: int) -> List[dict]:
    conn = _get_db()
    rows = conn.execute("""
        SELECT c.id, c.title, c.created_at, c.updated_at,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
        FROM conversations c
        WHERE c.user_id = ?
        ORDER BY c.updated_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_conversation(conv_id: str, user_id: int) -> bool:
    conn = _get_db()
    row = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, user_id)
    ).fetchone()
    if row is None:
        conn.close()
        return False
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return True


def get_messages(conv_id: str, user_id: int) -> Optional[List[dict]]:
    conn = _get_db()
    conv = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, user_id)
    ).fetchone()
    if conv is None:
        conn.close()
        return None
    rows = conn.execute("""
        SELECT id, role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC
    """, (conv_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_message(conv_id: str, user_id: int, role: str, content: str) -> Optional[dict]:
    if role not in ("user", "assistant", "system"):
        return None
    conn = _get_db()
    conv = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, user_id)
    ).fetchone()
    if conv is None:
        conn.close()
        return None
    msg_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content) VALUES (?, ?, ?, ?)",
        (msg_id, conv_id, role, content)
    )
    conn.execute(
        "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
        (conv_id,)
    )
    conn.commit()
    row = conn.execute(
        "SELECT id, role, content, created_at FROM messages WHERE id = ?",
        (msg_id,)
    ).fetchone()
    conn.close()
    return dict(row)

init_db()