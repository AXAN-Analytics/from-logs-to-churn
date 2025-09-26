import os
import uuid
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import Response

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel


APP_NAME = os.getenv("APP_NAME", "dummy-site")


app = FastAPI(title=f"{APP_NAME}")


# --- in-memory store just for demo ---
EVENTS: List[dict] = []


PAGE_TEMPLATE = """
    <!doctype html>
    <html>
    <head><title>{title}</title></head>
    <body>
    <h1>{title}</h1>
    <p>{body}</p>
    <script>
    // Simple beacon on page load
    fetch("/collect", {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{
                event_id: crypto.randomUUID(),
                event_type: "page_view",
                session_id: localStorage.session_id || (localStorage.session_id = crypto.randomUUID()),
                user_id: localStorage.user_id || (localStorage.user_id = crypto.randomUUID()),
                path: "{path}",
                ts: new Date().toISOString()
    }})
    }});
    </script>
    </body>
    </html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE_TEMPLATE.format(title="Home", body="Welcome to the dummy shop.", path="/")


@app.get("/product/{pid}", response_class=HTMLResponse)
def product(pid: str):
    return PAGE_TEMPLATE.format(title=f"Product {pid}", body=f"Details for product {pid}.", path=f"/product/{pid}")


@app.get("/login", response_class=HTMLResponse)
def login():
    return PAGE_TEMPLATE.format(title="Login", body="Pretend login page.", path="/login")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
class EventIn(BaseModel):
        event_id: str
        session_id: str
        user_id: str
        event_type: str
        ts: Optional[str] = None
        path: Optional[str] = None


@app.post("/collect")
async def collect(ev: EventIn, req: Request):
    enriched = ev.dict()
    enriched["received_at"] = datetime.now(timezone.utc).isoformat()
    enriched["ip"] = req.client.host if req.client else None
    EVENTS.append(enriched)
    # TODO: if DATABASE_URL present, buffer + bulk insert to Postgres
    return JSONResponse({"ok": True})


@app.get("/events")
def list_events(limit: int = 50):
    return JSONResponse(EVENTS[-limit:])


@app.get("/healthz")
def health():
    return PlainTextResponse("ok")