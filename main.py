import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# 2026 Standard ADK Imports
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

app = FastAPI(title="MIA - Final ADK Stable")

# 1. Global Session Service
session_service = InMemorySessionService()

# 2. Agent Definition
mia_agent = Agent(
    name="MIA_Meeting_Agent",
    model="gemini-2.0-flash", 
    instruction="Extract action items, decisions, and blockers from meetings. Return valid JSON."
)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None),
    session_id: str = Form("default_session")
):
    # Data Extraction
    content = ""
    if transcript_file and transcript_file.filename:
        content = (await transcript_file.read()).decode("utf-8")
    elif transcript_text:
        content = transcript_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    try:
        # ✅ THE CRITICAL FIX:
        # In ADK 1.2+, the Runner 'owns' the session. 
        # All IDs must be passed here, NOT in the .run() method.
        runner = Runner(
            app_name="MIA_App",
            agent=mia_agent, 
            session_service=session_service,
            session_id=session_id,  # Moved from .run()
            user_id="default_user"  # Moved from .run()
        )
        
        # Ensure session exists in memory
        await session_service.create_session(
            app_name="MIA_App", 
            user_id="default_user", 
            session_id=session_id
        )

        # ✅ THE CALL FIX:
        # .run() now takes exactly ONE positional argument: your prompt.
        prompt = f"Return a JSON analysis of this transcript: {content}"
        result = await runner.run(prompt)
        
        # Extract and Clean JSON
        raw_text = result.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()

        return json.loads(raw_text)
        
    except Exception as e:
        print(f"Final Debug Trace: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ADK Runner Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)