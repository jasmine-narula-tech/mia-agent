import os
import json
import asyncio
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import google.generativeai as genai

app = FastAPI(title="MIA - Meeting Intelligence Agent")

# API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY not set")

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash",
    generation_config={
        "temperature": 0.2
    }
)

# Split text
def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Run blocking AI call in thread (IMPORTANT)
async def summarize_chunk(chunk: str) -> str:
    def call_model():
        prompt = f"Summarize this meeting transcript in 2-3 sentences:\n{chunk}"
        response = model.generate_content(prompt)
        return response.text

    return await asyncio.to_thread(call_model)

@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None)
):
    content = ""

    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    elif transcript_text:
        content = transcript_text

    if not content.strip():
        raise HTTPException(status_code=400, detail="No transcript provided")

    # 1. Chunking
    chunks = chunk_text(content)

    # 2. Parallel summaries (SAFE now)
    summaries = await asyncio.gather(*[summarize_chunk(c) for c in chunks])
    combined_summary = "\n".join(summaries)

    # 3. Final structured output
    def final_call():
        prompt = f"""
        Analyze this meeting and return ONLY valid JSON.

        Transcript summary:
        {combined_summary}

        JSON:
        {{
            "summary": "string",
            "tone": "Positive / Neutral / Negative",
            "productivity_score": 1-10,
            "engagement": {{"speaker": "percent"}},
            "blockers": ["string"],
            "action_items": [
                {{"task": "string", "assigned_to": "string", "deadline": "string"}}
            ],
            "decisions": ["string"]
        }}
        """
        response = model.generate_content(prompt)
        return response.text

    try:
        result_text = await asyncio.to_thread(final_call)
        return json.loads(result_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")