import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# Correct ADK Imports for 2026
from google.adk.agents import Agent
from google.adk.runners import Runner

app = FastAPI(title="MIA - ADK Runner Edition")

# ✅ Define the Agent (The configuration)
mia_agent = Agent(
    name="MIA_Meeting_Agent",
    model="gemini-2.0-flash", 
    instruction=(
        "You are an expert meeting analyst. Extract action items, decisions, "
        "and blockers. Return results in strict JSON format. "
        "Ensure the JSON matches the schema provided in the prompt."
    )
)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None),
    session_id: str = Form("default_session")
):
    # 1. Extract content from Form or File
    content = ""
    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    elif transcript_text and transcript_text.strip():
        content = transcript_text

    # 2. Validation
    if not content.strip():
        raise HTTPException(status_code=400, detail="No transcript provided.")

    try:
        # ✅ THE ADK FIX: Use the Runner to execute the agentic loop
        # The Runner manages the orchestration, tools, and history
        runner = Runner()
        
        # We pass the content as user_input to the runner
        result = await runner.run(
            agent=mia_agent,
            user_input=f"Analyze this transcript: {content}",
            session_id=session_id
        )
        
        # 3. Clean and parse the response
        # ADK result.text often contains markdown code blocks (```json ... ```)
        raw_text = result.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1).replace("```", "", 1).strip()
        elif raw_text.startswith("```"):
             raw_text = raw_text.replace("```", "", 2).strip()

        return json.loads(raw_text)
        
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {result.text}")
        raise HTTPException(status_code=500, detail="Agent returned invalid JSON format.")
    except Exception as e:
        print(f"ADK Runner Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Agent Error: {str(e)}")

if __name__ == "__main__":
    # Cloud Run provides the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)