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
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    # 兼容旧数据库：如果 role 列不存在则添加
    try:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # 列已存在
    # 确保现有 admin 用户角色正确
    conn.execute("UPDATE users SET role = 'admin' WHERE username = 'admin' AND (role IS NULL OR role = '')")
    conn.commit()
    conn.close()


def _ensure_admin_exists():
    """确保存在 admin 管理员账户（如没有则创建）"""
    conn = _get_db()
    try:
        row = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
        if row is None:
            password_hash = _hash_password("admin")
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'admin')",
                ("admin", password_hash)
            )
            conn.commit()
            print("[INIT] 管理员账户已创建：admin / admin（请尽快修改密码）")
    finally:
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
    # 不允许注册 admin 用户名
    if username.lower() == "admin":
        return {"success": False, "message": "该用户名不允许注册"}

    conn = _get_db()
    try:
        password_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, 'user')",
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
            "SELECT id, password_hash, role FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if row is None:
            return {"success": False, "message": "用户名或密码错误"}

        if _hash_password(password) != row["password_hash"]:
            return {"success": False, "message": "用户名或密码错误"}

        token = _create_jwt(row["id"], username, row["role"])
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


def _create_jwt(user_id: int, username: str, role: str = "user") -> str:
    secret = _get_jwt_secret()
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
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
    return {
        "user_id": payload["sub"],
        "username": payload.get("username", "unknown"),
        "role": payload.get("role", "user")
    }


def get_all_users() -> list:
    """获取所有用户列表（仅管理员可用）"""
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM users ORDER BY id"
        ).fetchall()
        return [{
            "id": r["id"],
            "username": r["username"],
            "role": r["role"],
            "created_at": r["created_at"]
        } for r in rows]
    finally:
        conn.close()


def update_user_role(user_id: int, new_role: str) -> dict:
    """更新用户角色"""
    if new_role not in ("user", "admin"):
        return {"success": False, "message": "无效的角色"}
    conn = _get_db()
    try:
        row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return {"success": False, "message": "用户不存在"}
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
        conn.commit()
        return {"success": True, "message": f"用户ID {user_id} 角色已更新为 {new_role}"}
    finally:
        conn.close()


def delete_user(user_id: int) -> dict:
    """删除用户"""
    conn = _get_db()
    try:
        row = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return {"success": False, "message": "用户不存在"}
        if row["username"] == "admin":
            return {"success": False, "message": "不能删除超级管理员"}
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return {"success": True, "message": f"用户 {row['username']} 已删除"}
    finally:
        conn.close()


init_db()
_ensure_admin_exists()