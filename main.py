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
        # 1. Initialize Runner with the Agent and Service
        runner = Runner(
            agent=mia_agent, 
            session_service=session_service
        )
        
        # 2. Ensure the session is initialized in the service
        # This is required for the Runner to track the conversation
        await session_service.create_session(
            app_name="MIA_App", 
            user_id="default_user", 
            session_id=session_id
        )

        # ✅ THE FIX: Pass content as the first positional argument
        # In the latest ADK, 'user_input' keyword is replaced by a direct string
        result = await runner.run(
            f"Analyze this meeting transcript and return the JSON: {content}",
            session_id=session_id,
            user_id="default_user"
        )
        
        # 3. Handle the JSON extraction
        raw_text = result.text.strip()
        # Remove markdown if the model wrapped it
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()

        return json.loads(raw_text)
        
    except json.JSONDecodeError as je:
        print(f"JSON Error: {result.text}")
        raise HTTPException(status_code=500, detail="Agent failed to produce valid JSON.")
    except Exception as e:
        print(f"ADK Runner Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)