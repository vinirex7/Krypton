"""Controle manual seguro do processo live do Krypton.

O comando CLI não cria uma segunda instância do bot. Ele apenas grava um trigger
atômico; a instância já rodando em screen consome o pedido e executa o mesmo
`daily_cycle()` usado no agendamento normal.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import AGGRESSIVE_C_STATE_DB_FILE

TRIGGER_TTL_SECONDS = 600


def decision_trigger_path() -> Path:
    return Path(f"{AGGRESSIVE_C_STATE_DB_FILE}.decide-now")


def request_decision_now() -> Path:
    path = decision_trigger_path()
    payload = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def consume_decision_now(*, now: datetime | None = None, ttl_seconds: int = TRIGGER_TTL_SECONDS) -> dict | None:
    """Consome no máximo uma solicitação válida; pedidos antigos são descartados."""
    path = decision_trigger_path()
    try:
        raw = path.read_text(encoding="utf-8")
        path.unlink()
    except FileNotFoundError:
        return None

    try:
        payload = json.loads(raw)
        requested_at = datetime.fromisoformat(payload["requested_at"])
        if requested_at.tzinfo is None:
            requested_at = requested_at.replace(tzinfo=timezone.utc)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None

    current = now or datetime.now(timezone.utc)
    age = (current - requested_at.astimezone(timezone.utc)).total_seconds()
    if age < -5 or age > ttl_seconds:
        return None
    payload["age_seconds"] = max(0.0, age)
    return payload
