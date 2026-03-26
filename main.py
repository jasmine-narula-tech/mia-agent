from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json

app = FastAPI()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

class MeetingRequest(BaseModel):
    transcript: str

@app.get("/")
def home():
    return {"message": "MIA is running 🚀"}

@app.post("/analyze-meeting")
def analyze(req: MeetingRequest):
    prompt = f"""
    Analyze this meeting transcript and return ONLY valid JSON.

    Do NOT add any explanation or text.

    {{
      "summary": "string",
      "action_items": [],
      "decisions": []
    }}

    Transcript:
    {req.transcript}
    """

    response = model.generate_content(prompt)

    output = response.text.strip()

    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        return json.loads(json_str)
    except Exception as e:
        return {
            "error": "Failed to parse JSON",
            "raw_output": output
        }