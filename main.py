import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# Correct ADK Imports for 2026
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService  # <--- NEW IMPORT

app = FastAPI(title="MIA - ADK Runner Edition")

# 1. Initialize the Session Service globally
# This acts as the "database" for your current session
session_service = InMemorySessionService()

# 2. Define the Agent
mia_agent = Agent(
    name="MIA_Meeting_Agent",
    model="gemini-2.0-flash", 
    instruction=(
        "You are an expert meeting analyst. Extract action items, decisions, "
        "and blockers. Return results in strict JSON format."
    )
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
    content = ""
    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    elif transcript_text:
        content = transcript_text

    if not content:
        raise HTTPException(status_code=400, detail="No transcript provided.")

    try:
        # ✅ THE POSITIONING FIX:
        # We define the session context INSIDE the Runner initialization
        runner = Runner(
            app_name="MIA_Meeting_Assistant",
            agent=mia_agent, 
            session_service=session_service,
            session_id=session_id,  # Move session_id here
            user_id="default_user"  # Move user_id here
        )
        
        # Ensure session existence
        await session_service.create_session(
            app_name="MIA_Meeting_Assistant", 
            user_id="default_user", 
            session_id=session_id
        )

        # ✅ THE CALL FIX:
        # Now runner.run() only takes ONE positional argument (the prompt)
        result = await runner.run(
            f"Analyze this meeting and return valid JSON: {content}"
        )
        
        # Clean the JSON output
        raw_text = result.text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()

        return json.loads(raw_text)
        
    except Exception as e:
        print(f"ADK Final Trace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)