"""
用户管理模块 - 注册、登录、JWT 认证
使用标准库实现，无需额外依赖
"""
import os
import sqlite3
import hashlib
import hmac
import json
import base64
import time
import uuid
from typing import Optional

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "rag_users.db")

_JWT_SECRET = None

def _get_jwt_secret():
    global _JWT_SECRET
    if _JWT_SECRET is None:
        secret_path = os.path.join(DB_DIR, ".jwt_secret")
        if os.path.exists(secret_path):
            with open(secret_path, "r") as f:
                _JWT_SECRET = f.read().strip()
        else:
            _JWT_SECRET = uuid.uuid4().hex + uuid.uuid4().hex
            with open(secret_path, "w") as f:
                f.write(_JWT_SECRET)
    return _JWT_SECRET


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> dict:
    if not username or not password:
        return {"success": False, "message": "用户名和密码不能为空"}
    if len(username) < 2 or len(username) > 50:
        return {"success": False, "message": "用户名长度需在2-50字符之间"}
    if len(password) < 4:
        return {"success": False, "message": "密码长度不能少于4位"}

    conn = _get_db()
    try:
        password_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash)
        )
        conn.commit()
        return {"success": True, "message": f"用户 {username} 注册成功"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"用户名 '{username}' 已存在"}
    finally:
        conn.close()


def login_user(username: str, password: str) -> dict:
    if not username or not password:
        return {"success": False, "message": "用户名和密码不能为空"}

    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if row is None:
            return {"success": False, "message": "用户名或密码错误"}

        if _hash_password(password) != row["password_hash"]:
            return {"success": False, "message": "用户名或密码错误"}

        token = _create_jwt(row["id"], username)
        return {"success": True, "message": "登录成功", "token": token}
    finally:
        conn.close()


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _create_jwt(user_id: int, username: str) -> str:
    secret = _get_jwt_secret()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + 7 * 24 * 3600
    }
    header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("utf-8"),
        hashlib.sha256
    ).digest()
    signature_b64 = _base64url_encode(signature)
    return f"{signing_input}.{signature_b64}"


def verify_jwt(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, signature_b64 = parts
        secret = _get_jwt_secret()
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signing_input.encode("utf-8"),
            hashlib.sha256
        ).digest()
        actual_sig = _base64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_base64url_decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def get_current_user(authorization: Optional[str]) -> Optional[dict]:
    if not authorization:
        return None
    token = authorization
    if token.startswith("Bearer "):
        token = token[7:]
    payload = verify_jwt(token)
    if payload is None:
        return None
    return {"user_id": payload["sub"], "username": payload.get("username", "unknown")}


init_db()