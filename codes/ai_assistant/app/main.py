# -*- coding: utf-8 -*-
"""FastAPI 入口：会话管理、文档上传、RAG 流式问答、静态前端托管。

启动（在 codes/ai_assistant 目录下）：
    uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
然后浏览器打开 http://127.0.0.1:8000
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engine import engine

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.startup()
    yield
    await engine.shutdown()


app = FastAPI(title="AgentScope RAG 问答助手", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    message: str


def sse(event: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


# ---------------- 会话 ----------------
@app.post("/api/sessions")
async def create_session():
    sess = engine.create_session()
    return {"sid": sess.sid, "title": sess.title, "created_at": sess.created_at}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": engine.list_sessions()}


@app.get("/api/sessions/{sid}/messages")
async def get_messages(sid: str):
    sess = engine.get_session(sid)
    if sess is None:
        raise HTTPException(404, "会话不存在")
    return {"sid": sid, "title": sess.title, "messages": sess.messages}


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    ok = engine.delete_session(sid)
    if not ok:
        raise HTTPException(404, "会话不存在")
    return {"ok": True}


# ---------------- 流式问答（SSE） ----------------
@app.post("/api/sessions/{sid}/chat")
async def chat(sid: str, body: ChatIn):
    sess = engine.get_session(sid)
    if sess is None:
        raise HTTPException(404, "会话不存在")
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "消息为空")

    async def gen():
        try:
            async for event, data in engine.chat_stream(sess, message):
                yield sse(event, data)
        except Exception as exc:  # 兜底：任何异常都以一帧 error 结束
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------- 知识库 / 上传 ----------------
@app.post("/api/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        raw = await f.read()
        try:
            info = await engine.add_document(f.filename, raw)
        except ValueError as exc:
            raise HTTPException(415, str(exc)) from exc
        results.append(info)
    return {"documents": results}


@app.get("/api/documents")
async def list_documents():
    return {"documents": await engine.list_documents()}


@app.get("/api/health")
async def health():
    return {"ok": True}


# 静态前端放最后，避免吞掉 /api 路由
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
