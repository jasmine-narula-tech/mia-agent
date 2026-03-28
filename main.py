import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps import App
from google.genai import types

# DEFINITIONS AT TOP LEVEL 
MIA_APP_NAME = "MIA_Manager" 
session_service = InMemorySessionService()

# Define the Agent with the Strict Tone and Schema
mia_agent = Agent(
    name="MIA_Meeting_Agent",
    model="gemini-2.5-flash-lite", 
    instruction=(
        "You are an expert meeting analyst. Return ONLY a JSON object. "
        "RULES:"
        "1. 'productivity_score' MUST be a single-digit integer from 1 to 10 (e.g., 8)."
        "2. 'tone' MUST be one of: 'Positive', 'Negative', or 'Neutral'."
        "3. 'decisions' MUST be a simple list of strings (e.g., ['Decision A', 'Decision B'])."
        "4. 'summary' must be exactly 2 sentences."
        "Strict Schema:"
        "{"
        "  'summary': 'string',"
        "  'tone': 'Positive' | 'Negative' | 'Neutral',"
        "  'productivity_score': 8,"
        "  'engagement': {'Name': 'Level'},"
        "  'action_items': [{'task': 'string', 'assigned_to': 'string', 'deadline': 'string'}],"
        "  'decisions': ['string'],"
        "  'blockers': ['string']"
        "}"
    )
)

# Initialize the App object
mia_app = App(name=MIA_APP_NAME, root_agent=mia_agent)

app = FastAPI(title="MIA - Final Polish")


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    # Ensure index.html is in the same directory
    with open("index.html", "r") as f:
        return f.read()

@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None),
    session_id: str = Form("default_session")
):
    content = transcript_text or ""
    if transcript_file and transcript_file.filename:
        content = (await transcript_file.read()).decode("utf-8")

    if not content.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    try:
        # Handle existing sessions gracefully
        try:
            await session_service.create_session(
                app_name=MIA_APP_NAME, 
                user_id="default_user", 
                session_id=session_id
            )
        except Exception as e:
            if "already exists" in str(e):
                pass 
            else:
                raise e

        runner = Runner(app=mia_app, session_service=session_service)
        
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=f"Analyze: {content}")]
        )

        final_text = ""
        async for event in runner.run_async(
            user_id="default_user",
            session_id=session_id,
            new_message=user_message
        ):
            if event.is_final_response():
                final_text = event.content.parts[0].text

        # Clean JSON and Return
        raw_text = final_text.strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()

        return json.loads(raw_text)
        
    except Exception as e:
        print(f"ADK Debug: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)