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
    # 1. Get the content
    content = transcript_text or ""
    if transcript_file:
        content = (await transcript_file.read()).decode("utf-8")

    if not content:
        raise HTTPException(status_code=400, detail="No transcript provided.")

    try:
        # 2. Use ADK to run the agent
        # ADK handles the prompt construction and model interaction
        result = await mia_agent.run(content, session_id=session_id)
        
        # 3. Return the response text (ADK agents return a Response object)
        return json.loads(result.text)
        
    except Exception as e:
        print(f"ADK Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)