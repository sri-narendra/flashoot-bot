import os
import json
import uuid
from typing import Dict, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from google import genai
from google.genai import types

# FAISS + embeddings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# -----------------------------------------------------------------------------
# Load Structured Data
# -----------------------------------------------------------------------------
DATA_FILE = "data.json"

if not os.path.exists(DATA_FILE):
    raise RuntimeError("data.json not found")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

if not isinstance(raw_data, list) or not raw_data:
    raise RuntimeError("data.json must be a non-empty list")

# Extract contents for FAISS
documents = [item["content"] for item in raw_data]

# -----------------------------------------------------------------------------
# Embeddings + FAISS
# -----------------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedder.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# -----------------------------------------------------------------------------
# Session Memory
# -----------------------------------------------------------------------------
sessions: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY = 6

# -----------------------------------------------------------------------------
# Hybrid Retrieval (Keyword + FAISS)
# -----------------------------------------------------------------------------
def get_context(query: str, k: int = 3) -> str:
    query_lower = query.lower()

    # Step 1: Keyword filtering
    keyword_matches = []
    for item in raw_data:
        for kw in item.get("keywords", []):
            if kw in query_lower:
                keyword_matches.append(item)
                break

    # Optional: sort by priority
    keyword_matches = sorted(
        keyword_matches,
        key=lambda x: x.get("priority", 0),
        reverse=True
    )

    # Step 2: If keyword matches found → use them
    if keyword_matches:
        texts = [item["content"] for item in keyword_matches[:k]]
        return "\n".join(texts)

    # Step 3: Fallback to FAISS semantic search
    query_vec = embedder.encode([query])
    query_vec = np.array(query_vec).astype("float32")

    distances, indices = index.search(query_vec, k)

    results = [documents[i] for i in indices[0]]
    return "\n".join(results)

# -----------------------------------------------------------------------------
# Prompt Builder
# -----------------------------------------------------------------------------
def build_prompt(user_msg: str, history: List[Dict], context: str) -> str:
    history_text = ""
    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    return f"""
You are a student support assistant.

Rules:
- Keep answers under 120 words
- Be clear, practical, not overly emotional
- Use provided context when relevant
- If serious distress, suggest professional help

Relevant Information:
{context}

Conversation:
{history_text}

User: {user_msg}
Assistant:
"""

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_msg = (data.get("message") or "").strip()
    session_id = data.get("session_id")

    if not user_msg:
        return jsonify({"error": "Message required"}), 400

    # Create session
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Get relevant context
    context = get_context(user_msg)

    # Build prompt
    prompt = build_prompt(user_msg, history, context)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200,
            )
        )

        reply = (response.text or "").strip()

    except Exception as e:
        return jsonify({"error": "AI request failed"}), 500

    # Save memory
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": reply})

    sessions[session_id] = history[-MAX_HISTORY:]

    return jsonify({
        "reply": reply,
        "session_id": session_id
    })

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
