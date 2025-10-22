import os
import json
import logging
import re
from typing import List, Optional, Literal, Any, Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# 0) Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("context-service")

# -----------------------------
# 1) Env & OpenAI configuration
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Put it in .env or set in your shell.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# 2) Pydantic data contracts
# -----------------------------
class Message(BaseModel):
    role: Literal["user", "bot", "system"]
    message: str

class Slots(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    date: Optional[str] = None
    passengers: Optional[int] = None

class StructuredContext(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    intent: Optional[str] = None
    slots: Slots = Field(default_factory=Slots)
    flight_number: Optional[str] = None
    baggage_id: Optional[str] = None
    user_goal: Optional[str] = None
    context_summary: str
    history: List[Message]
    timestamp: Optional[str] = None

class ContextRequest(BaseModel):
    history: List[Message] = Field(default_factory=list)
    new_message: str

# -----------------------------
# 3) Prompt builder
# -----------------------------
def build_prompt(history: List[Message], new_message: str, schema: Dict[str, Any]) -> Dict[str, str]:
    # System message with rules + embedded schema spec (model is in JSON mode so it must return JSON)
    system_instructions = f"""
You are the Context Management module for a production conversational AI system.

Return ONE JSON OBJECT that matches the provided schema exactly.
Rules:
- "context_summary" is MANDATORY (concise and factual).
- If any field cannot be inferred, set it to null (do NOT guess).
- Echo a normalized 'history' including all prior turns plus the new user message.
- Valid roles are "user", "bot", and "system". Do not invent new roles.
- Output must be valid JSON, no extra keys, no commentary, no markdown.

Output JSON Schema (keys/types to follow; additional keys are NOT allowed):
{json.dumps(schema, indent=2)}
""".strip()

    def fmt(m: Message) -> str:
        return f"{m.role.upper()}: {m.message}"

    convo = "\n".join([fmt(m) for m in history] + [f"USER: {new_message}"])

    user_content = f"""
Conversation:
{convo}
""".strip()

    return {"system": system_instructions, "user": user_content}

# -----------------------------
# 4) Heuristic extractor (debug fallback)
# -----------------------------
from datetime import datetime, date
from dateutil import parser as dparser
import re
from typing import Optional, Tuple, List

MONTH_WORDS = {
    "jan","january","feb","february","mar","march","apr","april","may","jun","june","jul","july",
    "aug","august","sep","sept","september","oct","october","nov","november","dec","december"
}
STOP_AFTER_DEST = {"on","for","please","at","by","around","about","and"}
DATE_TOKEN_RE = re.compile(r"^\d{1,2}(st|nd|rd|th)?$|^\d{4}$", re.I)

from datetime import datetime, date
from dateutil import parser as dparser
import re

INTENT_CANON = {
    "book_flight": "book_ticket",
    "book_ticket": "book_ticket",
    "flight_booking": "book_ticket",
}

MONTH_WORDS = {
    "jan","january","feb","february","mar","march","apr","april","may","jun","june","jul","july",
    "aug","august","sep","sept","september","oct","october","nov","november","dec","december"
}
STOP_TOKENS = {",", ".", "!", "?", ";", ":"}

def _parse_iso_date_loose(text: str | None) -> str | None:
    """Parse any natural date to ISO (YYYY-MM-DD). If year missing, assume next occurrence."""
    if not text or not text.strip():
        return None
    today = date.today()
    try:
        base = datetime(today.year, today.month, today.day)
        dt = dparser.parse(text, fuzzy=True, default=base)
        if dt.date() < today and not re.search(r"\b20\d{2}\b", text):
            dt = dt.replace(year=dt.year + 1)
        return dt.date().isoformat()
    except Exception:
        return None

def _clean_loc(s: str | None) -> str | None:
    if not s: return None
    s = s.strip()
    # Drop trailing stop tokens
    while s and s[-1] in STOP_TOKENS:
        s = s[:-1].strip()
    return s or None

def _to_int_or_none(v):
    if v is None: return None
    if isinstance(v, int): return v
    s = str(v).strip()
    if s.isdigit(): return int(s)
    words = {"one":1,"two":2,"three":3,"four":4,"five":5}
    return words.get(s.lower())

def normalize_structured_context(sc: "StructuredContext") -> "StructuredContext":
    # Canonicalize intent
    if sc.intent:
        sc.intent = INTENT_CANON.get(sc.intent.lower(), sc.intent)

    # Clean locations
    sc.slots.origin = _clean_loc(sc.slots.origin)
    sc.slots.destination = _clean_loc(sc.slots.destination)

    # Normalize date to ISO if possible
    iso = _parse_iso_date_loose(sc.slots.date)
    if iso:
        sc.slots.date = iso

    # Normalize passengers to int
    sc.slots.passengers = _to_int_or_none(sc.slots.passengers)

    # Compose user_goal if absent and we have enough info
    if not sc.user_goal:
        bits = []
        if sc.slots.origin: bits.append(f"from {sc.slots.origin}")
        if sc.slots.destination: bits.append(f"to {sc.slots.destination}")
        if sc.slots.date: bits.append(f"on {sc.slots.date}")
        if sc.slots.passengers: bits.append(f"for {sc.slots.passengers} passenger" + ("s" if sc.slots.passengers > 1 else ""))
        if bits:
            sc.user_goal = "Book flight " + " ".join(bits) + "."

    # Ensure context_summary is present and helpful
    if not sc.context_summary or not sc.context_summary.strip():
        if sc.user_goal:
            sc.context_summary = sc.user_goal
        else:
            sc.context_summary = "Conversation summary unavailable."

    return sc


def _normalize_date_guess(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    today = date.today()
    try:
        base = datetime(today.year, today.month, today.day)
        dt = dparser.parse(text, fuzzy=True, default=base)
        # If user didn’t specify a year and parsed date already passed this year, roll to next year
        if dt.date() < today and not re.search(r"\b20\d{2}\b", text):
            dt = dt.replace(year=dt.year + 1)
        return dt.date().isoformat()
    except Exception:
        return None

def _clean_location(loc: str) -> str:
    # Trim trailing stop words / month words / numeric date tokens (e.g., "Delhi October 25th" -> "Delhi")
    tokens = [t for t in re.split(r"\s+", loc.strip()) if t]
    cleaned = []
    for t in tokens:
        low = t.lower().strip(".,!?")
        if low in STOP_AFTER_DEST or low in MONTH_WORDS or DATE_TOKEN_RE.match(low):
            break
        cleaned.append(t.strip(",."))
    return " ".join(cleaned).strip()

def crude_extract(history: List["Message"], new_message: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    # Only user messages + the new user message
    user_text = " ".join([m.message for m in history if m.role == "user"] + [new_message])

    # Find all "from X to Y" pairs; take the last (closest to the latest message)
    pairs = re.findall(r"[Ff]rom\s+([A-Za-z][\w .-]{1,60})\s+(?:to|2)\s+([A-Za-z][\w .-]{1,80})", user_text)
    origin = destination = None
    if pairs:
        origin_raw, dest_raw = pairs[-1]
        origin = _clean_location(origin_raw)
        destination = _clean_location(dest_raw)

    # Passengers
    m2 = re.search(r"\bfor\s+(\d+|one|two|three|four|five)\s+(people|passengers)\b", user_text, re.I)
    word2num = {"one":1,"two":2,"three":3,"four":4,"five":5}
    passengers: Optional[int] = None
    if m2:
        v = m2.group(1).lower()
        passengers = int(v) if v.isdigit() else word2num.get(v)

    # Date
    date_iso = _normalize_date_guess(user_text)

    # Normalize empties to None
    if origin == "": origin = None
    if destination == "": destination = None

    return origin, destination, date_iso, passengers


# -----------------------------
# 5) OpenAI call (Chat Completions JSON mode)
# -----------------------------
def call_openai_json_mode(system_msg: str, user_msg: str) -> str:
    """
    Calls Chat Completions with JSON mode enabled. Returns JSON string.
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"}  # <-- JSON mode
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("Empty completion content.")
    return content

# -----------------------------
# 6) FastAPI app & endpoints
# -----------------------------
app = FastAPI(title="Context Management Service (OpenAI Chat JSON Mode)", version="1.1.0")

@app.post("/summarize_context", response_model=StructuredContext)
def summarize_context(req: ContextRequest) -> StructuredContext:
    full_history = list(req.history) + [Message(role="user", message=req.new_message)]
    schema = StructuredContext.model_json_schema()
    prompt = build_prompt(req.history, req.new_message, schema)

    try:
        raw_json = call_openai_json_mode(prompt["system"], prompt["user"])
        raw_obj = json.loads(raw_json)

        sc = StructuredContext.model_validate(raw_obj)
        # Ensure normalized history (we don’t trust the model for this)
        sc.history = full_history
        sc = normalize_structured_context(sc)

        if not sc.context_summary or not sc.context_summary.strip():
            sc.context_summary = f"User said: {req.new_message}"

        return sc

    except Exception as e:
        origin, dest, date_iso, pax = crude_extract(req.history, req.new_message)

        # Compose a clean, informative summary
        summary_bits = ["Book flight"]
        if origin: summary_bits.append(f"from {origin}")
        if dest:   summary_bits.append(f"to {dest}")
        if date_iso: summary_bits.append(f"on {date_iso}")
        if pax:    summary_bits.append(f"for {pax} passenger{'s' if pax and pax>1 else ''}")
        summary = " ".join(summary_bits) + "."

        sc = StructuredContext(
            user_id=None,
            session_id=None,
            intent="book_ticket" if (origin or dest or pax or date_iso) else None,
            slots=Slots(origin=origin, destination=dest, date=date_iso, passengers=pax),
            flight_number=None,
            baggage_id=None,
            user_goal=summary if (origin or dest or pax or date_iso) else None,
            context_summary=summary if (origin or dest or pax or date_iso) else f"User said: {req.new_message}",
            history=full_history,
            timestamp=None
        )
        sc = normalize_structured_context(sc)
        return sc




@app.get("/health")
def health():
    return {"status": "ok", "model": OPENAI_MODEL}
