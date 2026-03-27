import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# ✅ Use the official New SDK
from google import genai
from google.genai import types

app = FastAPI(title="MIA - Meeting Intelligent Assistant")

# API Key check
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Important: In Cloud Run, don't raise Exception at top level if possible, 
    # but for debugging it's fine.
    print("WARNING: GEMINI_API_KEY not set")

# ✅ Initialize Official Client
client = genai.Client(api_key=api_key)

# System Instruction for the AI
SYSTEM_INSTRUCTION = """
You are an AI meeting assistant.
Analyze meeting transcripts and return ONLY valid JSON in this format:

{
  "summary": "string",
  "tone": "Positive / Neutral / Negative",
  "productivity_score": 1-10,
  "engagement": {"speaker_name": "percent"},
  "blockers": ["string"],
  "action_items": [
    {
      "task": "string",
      "assigned_to": "string",
      "deadline": "string or null"
    }
  ],
  "decisions": ["string"]
}

Do NOT add explanations. Return ONLY JSON.
"""

# UI route
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>"

# Analyze endpoint
@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None)
):
    content = ""

    # 1. Read input
    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    elif transcript_text:
        content = transcript_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="No transcript provided")

    # 2. Call Gemini (Using official SDK structure)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", # Stable 2026 Lite model
            contents=f"Transcript:\n{content[:8000]}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.7
            )
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="No response from AI")

        # The SDK's response_mime_type ensures no markdown backticks are needed
        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

if __name__ == "__main__":
    # Cloud Run provides the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)