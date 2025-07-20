from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import generate_text
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to talk to backend locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate(prompt: Prompt):
    output = generate_text(prompt.prompt)
    return {"response": output}
