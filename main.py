import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse


# ADK import
from google.adk.agents import Agent

app = FastAPI(title="MIA - Meeting Intelligent Assistant")

# API Key check
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY not set")

# Create ADK Agent
agent = Agent(
    model="gemini-1.5-flash-lite", 
    description="Meeting Intelligence Agent",
    instruction="""
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
)

# Helper: clean markdown JSON
def clean_json(text: str):
    return text.replace("```json", "").replace("```", "").strip()

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

    # Read input
    if transcript_file and transcript_file.filename:
        if not transcript_file.filename.endswith(".txt"):
            raise HTTPException(status_code=400, detail="Only .txt files supported")
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")

    elif transcript_text:
        content = transcript_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="No transcript provided")

    # Limit size (fast response)
    content = content[:5000]

    prompt = f"""
    Transcript:
    {content}
    """

    try:
        # ✅ ADK Agent call
        response = agent.run(prompt)

        if not response:
            raise HTTPException(status_code=500, detail="No response from AI")

        cleaned = clean_json(response)

        return json.loads(cleaned)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)