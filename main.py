import os
import json
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import google.generativeai as genai

app = FastAPI(title="MIA - Meeting Intelligence Agent")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY not set")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="models/gemini-3-flash-preview",
    generation_config={
        "response_mime_type": "text",  # We'll parse JSON manually
        "temperature": 0.2             # Lower temp for structured output
    }
)

# Helper: Split transcript into chunks
def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

# Helper: Summarize a single chunk asynchronously
async def summarize_chunk(chunk: str) -> str:
    prompt = f"""
    Summarize this meeting transcript chunk in 2-3 concise sentences:
    {chunk}
    """
    response = await model.agenerate_content(prompt)
    return response.generations[0].content.strip()

# Optimized Analyze Endpoint
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

    # 1️⃣ Split transcript into chunks
    chunks = chunk_text(content, chunk_size=3000)

    # 2️⃣ Summarize each chunk in parallel
    summaries = await genai.asyncio.gather(*[summarize_chunk(c) for c in chunks])
    combined_summary = "\n".join(summaries)

    # 3️⃣ Generate final JSON
    final_prompt = f"""
    Analyze this summarized transcript and return ONLY valid JSON:

    Summarized Transcript:
    {combined_summary}

    JSON schema:
    {{
        "summary": "string",
        "tone": "Positive / Neutral / Negative",
        "productivity_score": 1-10,
        "engagement": {{"speaker_name": "percent of speaking time"}},
        "blockers": ["string"],
        "action_items": [
            {{"task": "string", "assigned_to": "string", "deadline": "string or null"}}
        ],
        "decisions": ["string"]
    }}
    """
    try:
        final_response = await model.agenerate_content(final_prompt)
        result_text = final_response.generations[0].content.strip()
        return json.loads(result_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")