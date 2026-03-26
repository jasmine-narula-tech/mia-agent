import os
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types

app = FastAPI(title="MIA - Meeting Intelligent Assistant")

# 1. Initialize the new GenAI Client
# It automatically looks for the GEMINI_API_KEY environment variable.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found</h1>"

@app.post("/analyze")
async def analyze_meeting(
    transcript_text: Optional[str] = Form(None),
    transcript_file: Optional[UploadFile] = File(None)
):
    # 1. Extract content from either the text area or the uploaded file
    content = ""
    if transcript_file and transcript_file.filename:
        file_bytes = await transcript_file.read()
        content = file_bytes.decode("utf-8")
    elif transcript_text:
        content = transcript_text
    
    if not content.strip():
        raise HTTPException(status_code=400, detail="No transcript provided")

    # 2. Call Gemini using the new SDK structure
    try:
        response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"Summarize this meeting transcript into JSON (summary, action_items, decisions): {content}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=1.0
        )
    )
        
        # In the new SDK, access text directly via response.text
        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")