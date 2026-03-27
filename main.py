import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import google.generativeai as genai

app = FastAPI(title="MIA - Meeting Intelligent Assistant")

# ✅ API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY not set")

genai.configure(api_key=api_key)

# ✅ Use your WORKING model
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 0.2
    }
)

# ✅ UI route
@app.get("/", response_class=HTMLResponse)
async def get_ui():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>"

# ✅ Analyze endpoint (TEXT + FILE)
@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None)
):
    content = ""

    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    elif transcript_text:
        content = transcript_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="No transcript provided")

    
    content = content[:5000]
    prompt = f"""
    Analyze this meeting transcript and return ONLY valid JSON in the specified schema.

    Transcript:
    {content}

    JSON schema:
    {{
    "summary": "string",
    "tone": "Positive / Neutral / Negative",
    "productivity_score": 1-10,
    "engagement": {{"speaker_name": "percent of speaking time"}},
    "blockers": ["string"],
    "action_items": [
        {{
        "task": "string",
        "assigned_to": "string",
        "deadline": "string or null"
        }}
    ],
    "decisions": ["string"]
    }}
    """

    try:
        response = model.generate_content(prompt)

        if not response.text:
            raise HTTPException(status_code=500, detail="No response from AI")

        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")