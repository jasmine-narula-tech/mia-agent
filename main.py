import os
import pydantic
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

app = FastAPI(title="MIA - Meeting Intelligence Agent")

# 1. New SDK Client Setup (Asynchronous)
# Note: It automatically picks up GEMINI_API_KEY from environment
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).aio

# 2. Define your output structure using Pydantic
class ActionItem(pydantic.BaseModel):
    task: str
    assigned_to: str
    deadline: str

class MeetingAnalysis(pydantic.BaseModel):
    summary: str
    tone: str
    productivity_score: int
    engagement: Dict[str, str]
    blockers: List[str]
    action_items: List[ActionItem]
    decisions: List[str]

# Model ID for 2026
MODEL_ID = "gemini-3-flash-preview"

def chunk_text(text: str, chunk_size: int = 8000) -> List[str]:
    # Gemini 3 has a huge context, but chunking is still good for parallel speed
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None)
):
    # 1. Get Content
    content = ""
    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    elif transcript_text:
        content = transcript_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="No transcript provided")

    try:
        # 2. Parallel Chunk Summarization (Using the new .aio client)
        chunks = chunk_text(content)
        
        # We define a helper task for the parallel calls
        async def get_summary(c):
            res = await client.models.generate_content(
                model=MODEL_ID,
                contents=f"Summarize this meeting segment briefly: {c}"
            )
            return res.text

        chunk_summaries = await asyncio.gather(*[get_summary(c) for c in chunks])
        combined_context = "\n".join(chunk_summaries)

        # 3. Final Structured Analysis
        # The new SDK handles the JSON schema generation and parsing for you!
        response = await client.models.generate_content(
            model=MODEL_ID,
            contents=f"Perform a deep analysis of this meeting summary: {combined_context}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MeetingAnalysis, # Directly pass the Pydantic class
                temperature=0.7
            )
        )

        # 4. Return the parsed object directly
        return response.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MIA Agent Error: {str(e)}")

# Run with: python3 -m uvicorn main:app --reload --port 8080