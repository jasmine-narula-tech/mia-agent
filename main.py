import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# Stable 2026 ADK Imports
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App  # <--- Added this to fix "Session not found"
from google.genai import types

app = FastAPI(title="MIA - Zero Error Edition")

# 1. Global Session Service
session_service = InMemorySessionService()

# 2. Define the Agent
mia_agent = Agent(
    name="MIA_Meeting_Agent",
    model="gemini-2.5-flash-lite", 
    instruction=(
        "You are an expert meeting analyst. Analyze the transcript and return ONLY a JSON object "
        "with this exact structure: "
        "{"
        "  'summary': 'A 2-sentence overview',"
        "  'tone': 'Professional/Casual/Tense',"
        "  'productivity_score': 1-10 integer,"
        "  'engagement': {'PersonName': 'High/Med/Low'},"
        "  'action_items': [{'task': '...', 'assigned_to': '...', 'deadline': '...'}],"
        "  'decisions': ['Decision 1', 'Decision 2'],"
        "  'blockers': ['Risk 1']"
        "}"
    )
)

# 3. Wrap in an App (This prevents the 'Session Not Found' mismatch)
# The App name MUST match the name used in your session creation
MIA_APP_NAME = "MIA_Manager"
mia_app = App(name=MIA_APP_NAME, root_agent=mia_agent)

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
        content = (await transcript_file.read()).decode("utf-8")
    elif transcript_text:
        content = transcript_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    try:
        # ✅ STEP 1: Create the session and WAIT for it
        # This ensures the ID exists in the service before the runner looks for it
        await session_service.create_session(
            app_name=MIA_APP_NAME, 
            user_id="default_user", 
            session_id=session_id
        )

        # ✅ STEP 2: Initialize Runner with the APP object
        runner = Runner(
            app=mia_app,  # Using 'app' instead of 'agent' fixes the lookup error
            session_service=session_service
        )

        # ✅ STEP 3: Format the message correctly
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=f"Analyze this transcript and return JSON: {content}")]
        )

        # ✅ STEP 4: Run and collect response
        final_text = ""
        async for event in runner.run_async(
            user_id="default_user",
            session_id=session_id,
            new_message=user_message
        ):
            if event.is_final_response():
                final_text = event.content.parts[0].text

        # Standard JSON cleaning
        raw_text = final_text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()

        return json.loads(raw_text)
        
    except Exception as e:
        print(f"ADK Debug: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Runner Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)