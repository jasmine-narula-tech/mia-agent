from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json

app = FastAPI()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

class MeetingRequest(BaseModel):
    transcript: str

@app.get("/")
def home():
    return {"message": "MIA is running 🚀"}

@app.post("/analyze-meeting")
def analyze(req: MeetingRequest):
    prompt = f"""
    Analyze this meeting transcript and return JSON:

    {{
      "summary": "...",
      "action_items": [],
      "decisions": []
    }}

    Transcript:
    {req.transcript}
    """

    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except:
        return {"error": response.text}
