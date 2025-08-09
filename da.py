# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "httpx",
# ]
# ///

import os
import re
import sys
import asyncio
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import subprocess

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_KEY")
MODEL_CODE = "qwen/qwen3-coder"
MODEL_CONTEXT = "google/gemini-2.5-pro"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

async def ask_qwen_async(messages):
    """Send messages to Qwen3 Coder and return code output asynchronously."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(OPENROUTER_URL, headers=HEADERS, json={
            "model": MODEL_CODE,
            "messages": messages,
        })
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

def extract_code(text):
    """Extract Python code block from markdown."""
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            if parts[i].strip().startswith("python"):
                return parts[i].strip()[6:].strip()
            elif not parts[i].strip().startswith(("json", "text")):
                return parts[i].strip()
    return text.strip()

async def run_code_with_llm(messages):
    """Ask LLM for code and execute it without writing to disk."""
    iteration = 1
    while True:
        logger.info(f"=== Iteration {iteration} ===")
        iteration += 1

        llm_output = await ask_qwen_async(messages)
        code = extract_code(llm_output)

        # Run code in subprocess without saving to disk
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            final_output = stdout.decode().strip()
            logger.info("Code executed successfully.")
            return final_output
        else:
            error_message = stderr.decode().strip()
            logger.error(f"Error occurred: {error_message}")

            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": f"The code failed with this error:\n{error_message}\nPlease fix it and return the full corrected code."
            })

async def task_breakdown(task: str):
    """Break down the task using LLM."""
    with open("llm_task_breakdown_prompt.md", "r", encoding="utf-8") as file:
        prompt_for_breaking = file.read()

    full_prompt = f"{task}\n\n{prompt_for_breaking}"
    logger.info("Prompt sent for task breakdown")

    payload = {
        "model": MODEL_CONTEXT,
        "messages": [
            {"role": "system", "content": "You are an expert python developer that breaks down given statements into programming tasks."},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(OPENROUTER_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        logger.info("Task breakdown completed")
        return result

@app.post("/api/")
async def upload_files(request: Request):
    async def process_request():
        logger.info("Starting request processing")
        form = await request.form()
        files_received = []
        questions_content = None
        file_content_map = {}

        BASE_DIR = Path(__file__).resolve().parent
        files_dir = BASE_DIR / "files"
        files_dir.mkdir(exist_ok=True)

        # Save files to disk if needed, but process content in memory
        for field_name, field_value in form.items():
            if hasattr(field_value, "filename"):
                content_bytes = await field_value.read()
                try:
                    content_text = content_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content_text = "[binary content not shown]"
                file_content_map[field_value.filename] = content_text

                save_path = files_dir / field_value.filename
                with open(save_path, "wb") as f:
                    f.write(content_bytes)

                if field_value.filename == "questions.txt":
                    questions_content = content_text

                files_received.append({
                    "field_name": field_name,
                    "filename": field_value.filename,
                    "content_type": field_value.content_type,
                    "saved_to": str(save_path)
                })

        logger.info(f"Files received: {files_received}")

        if questions_content is None:
            raise HTTPException(status_code=400, detail="Missing required file: questions.txt")

        logger.info(f"Questions content: {questions_content[:100]}...")

        # Step 1: Break down tasks
        instruction_text = await task_breakdown(questions_content)

        messages = [
            {"role": "system", "content": "You are an expert Python coder and a data analyst. Always return only complete runnable Python code."},
            {"role": "user", "content": instruction_text}
        ]

        # Step 2: Generate and run code
        result = await run_code_with_llm(messages)

        return Response(content=result, media_type="application/json")

    try:
        return await asyncio.wait_for(process_request(), timeout=180)
    except asyncio.TimeoutError:
        return JSONResponse(content={"message": "Response from LLMs timed Out, please try again"}, status_code=504)
