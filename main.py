import os
import json
import uvicorn
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse

# Import ADK
from google.adk.agents import Agent

app = FastAPI(title="MIA - ADK Agent")

# Define the ADK Agent
# This is the "Brain" of your application
mia_agent = Agent(
    name="MIA_Meeting_Agent",
    model="gemini-2.0-flash", # Or gemini-3-flash if available in your project
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

    # 1. Check if a file was uploaded
    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    
    # 2. If no file, check the text area
    elif transcript_text and transcript_text.strip():
        content = transcript_text

    # 3. Final validation
    if not content.strip():
        # This is where your 400 error is coming from
        raise HTTPException(status_code=400, detail="No transcript provided.")

    try:
        # ADK Run
        result = await mia_agent.run(content, session_id=session_id)
        return json.loads(result.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)