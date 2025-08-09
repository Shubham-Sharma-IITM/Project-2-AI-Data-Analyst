# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "python-multipart",
#   "uvicorn",
#   "httpx",
# ]
# ///

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import httpx
import requests
import subprocess
import tempfile
import os
from fastapi import Response
import asyncio

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
INSTRUCTION_FILE = "breaked_task.txt"
MODEL_CODE = "qwen/qwen3-coder"
MODEL_CONTEXT = "google/gemini-2.5-pro"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

import sys
import re
import subprocess

# def install_dependencies_from_code(code: str):
#     """
#     Scan the code for imports and install all listed packages.
#     If a dependency is already installed, pip will skip it.
#     """
#     imports = set()

#     for line in code.splitlines():
#         line = line.strip()
#         if line.startswith("import "):
#             modules = [m.strip().split(".")[0] for m in line.replace(",", " ").split()[1:]]
#             imports.update(modules)
#         elif line.startswith("from "):
#             module = line.split(" ")[1].split(".")[0]
#             imports.add(module)

#     if imports:
#         print(f"Installing dependencies: {imports}")
#         subprocess.run([sys.executable, "-m", "pip", "install", *imports], check=False)
#     else:
#         print("No imports found in code.")



def ask_qwen(messages):
    """Send messages to Qwen3 Coder and return code output."""
    response = requests.post(OPENROUTER_URL, headers=HEADERS, json={
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

def run_code_with_llm(messages):
    iteration = 1
    while True:
        print(f"\n=== Iteration {iteration} ===")
        iteration += 1

        # Step 1: Ask LLM for code
        llm_output = ask_qwen(messages)
        code = extract_code(llm_output)

        print("\n--- Generated Code ---\n", code)

        # if iteration == 2:  # First loop after LLM output
        #     install_dependencies_from_code(code)

        # Step 2: Save to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        # Step 3: Try running the code
        try:
            result = subprocess.run(
                ["python", temp_path],
                check=True,
                capture_output=True,
                text=True
            )
            final_output = result.stdout.strip()
            print("Code executed successfully.")
            print("Final Output -", final_output)
            return final_output  # Exit on success

        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip()
            print("Error occurred:", error_message)

            # Step 4: Send error back to LLM
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({
                "role": "user",
                "content": f"The code failed with this error:\n{error_message}\nPlease fix it and return the full corrected code."
            })

        finally:
            os.remove(temp_path)


def task_breakdown(task: str):
    with open("llm_task_breakdown_prompt.md", "r", encoding="utf-8") as file:
        prompt_for_breaking = file.read()

    full_prompt = f"{task}\n\n{prompt_for_breaking}"
    print(full_prompt)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_CONTEXT,
        "messages": [
            {"role": "system", "content": "You are an expert python developer that breaks down given statements into programming tasks."},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0.3,
    }

    try:
        response = httpx.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]

        with open("breaked_task.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("Task breakdown completed")
        return result
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"AIPipe request failed: {str(e)}")

@app.post("/api/")
async def upload_files(request: Request):
    async def process_request():
        print("Starting:")
        form = await request.form()
        files_received = []
        questions_content = None
        file_content_map = {}
        print("Creating directory in base")
        BASE_DIR = Path(__file__).resolve().parent
        files_dir = BASE_DIR / "files"
        files_dir.mkdir(exist_ok=True)

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
        print(files_received)

        if questions_content is None:
            raise HTTPException(status_code=400, detail="Missing required file: questions.txt")
        print(questions_content)

        # Run task_breakdown in thread since it uses httpx.post (blocking)
        task_breakdown_result = await asyncio.to_thread(task_breakdown, questions_content)

        with open(INSTRUCTION_FILE, "r", encoding="utf-8") as f:
            instruction_text = f.read()
        print(instruction_text)

        messages = [
            {"role": "system", "content": "You are an expert Python coder and a data analyst. Always return only complete runnable Python code."},
            {"role": "user", "content": instruction_text}
        ]

        # run_code_with_llm is blocking, run in thread
        result = await asyncio.to_thread(run_code_with_llm, messages=messages)

        return Response(content=result, media_type="application/json")

    try:
        # Run the entire process_request with a 3-minute timeout
        return await asyncio.wait_for(process_request(), timeout=180)
    except asyncio.TimeoutError:
        return JSONResponse(content={"message": "Response from LLMs timed Out, please try again"}, status_code=504)