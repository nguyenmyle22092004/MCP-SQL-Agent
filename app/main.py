from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router

app = FastAPI(title="House Price SQL Agent")
app.include_router(router)
CHAT_UI_PATH = Path(__file__).parent / "web" / "chat.html"


@app.get("/")
def root():
    return FileResponse(CHAT_UI_PATH, media_type="text/html")
