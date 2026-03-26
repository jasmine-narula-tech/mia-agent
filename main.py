import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI(title="MIA - Meeting Intelligent Assistant")

# 1. API Configuration
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY environment variable")

genai.configure(api_key=api_key)

# 2. Updated Model (Gemini 3 Flash is the 2026 standard for speed/logic)
# Note: 'gemini-1.5-flash' is retired. 
MODEL_ID = "models/gemini-3-flash-preview" 

model = genai.GenerativeModel(
    model_name=MODEL_ID,
    generation_config={
        "response_mime_type": "application/json",
        "temperature": 1.0  # Recommended default for Gemini 3 series
    }
)

class MeetingRequest(BaseModel):
    transcript: str

@app.get("/")
def home():
    return {"message": "MIA is running on Gemini 3 🚀"}

@app.post("/analyze-meeting")
async def analyze(req: MeetingRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty")

    prompt = f"""
    Analyze this meeting transcript. Return a JSON object with:
    - summary (string)
    - action_items (list of strings)
    - decisions (list of strings)

    Transcript:
    {req.transcript}
    """

    try:
        response = model.generate_content(prompt)
        
        if not response.text:
            raise HTTPException(status_code=500, detail="AI returned no content")

        return json.loads(response.text)

    except Exception as e:
        # If the model name itself is the issue, this will catch it
        raise HTTPException(
            status_code=500, 
            detail=f"MIA Error: {str(e)}"
        )