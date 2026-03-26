import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# 1. Setup FastAPI
app = FastAPI(title="MIA - Meeting Intelligent Assistant")

# 2. Configure Gemini API
# Ensure your environment variable GEMINI_API_KEY is set
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY environment variable")
genai.configure(api_key=api_key)

# 3. Initialize Model with JSON constraints
# Using 'models/gemini-1.5-flash' explicitly to avoid NOT_FOUND errors
model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# 4. Define Data Models
class MeetingRequest(BaseModel):
    transcript: str

# 5. Routes
@app.get("/")
def home():
    return {"message": "MIA is running 🚀"}

@app.post("/analyze-meeting")
async def analyze(req: MeetingRequest):
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    prompt = f"""
    Analyze the following meeting transcript and return a JSON object.
    
    Structure:
    {{
      "summary": "A concise paragraph summarizing the meeting",
      "action_items": ["list", "of", "tasks"],
      "decisions": ["list", "of", "key", "decisions"]
    }}

    Transcript:
    {req.transcript}
    """

    try:
        # Generate content
        response = model.generate_content(prompt)
        
        # Check if response actually contains text
        if not response.text:
            raise HTTPException(status_code=500, detail="AI returned an empty response")

        # Parse the JSON string returned by Gemini
        # Since we used response_mime_type, this is guaranteed to be JSON
        return json.loads(response.text)

    except Exception as e:
        # This catches API errors, parsing errors, or safety filters
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing transcript: {str(e)}"
        )

# To run this, use: uvicorn main:app --reload