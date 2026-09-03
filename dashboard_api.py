"""Authenticated HTTP API for the Krypton client dashboard."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from telemetry import TelemetryStore

DB_PATH = os.getenv("KRYPTON_DASHBOARD_DB", "krypton_dashboard.db")
store = TelemetryStore(DB_PATH)
app = FastAPI(title="Krypton Dashboard API", version="1.0.0")


class LoginRequest(BaseModel):
    email: str
    password: str


def current_client(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.split(" ", 1)[1].strip()
    client_id = store.client_for_token(token)
    if not client_id:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return client_id


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/login")
def login(payload: LoginRequest) -> dict:
    auth = store.authenticate(payload.email, payload.password)
    if not auth:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token, client_id = auth
    return {"token": token, "client_id": client_id}


@app.get("/api/dashboard")
def dashboard(client_id: str = Depends(current_client)) -> dict:
    return store.dashboard(client_id)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = Path(__file__).resolve().parent / "dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return html_path.read_text(encoding="utf-8")
