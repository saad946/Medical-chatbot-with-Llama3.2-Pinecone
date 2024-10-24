from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from src.app import qa  # Ensure that this import path is correct
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Mounting the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/get")
async def chat(msg: str = Form(...)):
    result = qa({"query": msg})
    print("Input:", msg)
    print("Response:", result["result"])
    return {"response": result["result"]}