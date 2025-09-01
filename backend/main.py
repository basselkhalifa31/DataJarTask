from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import ollama
import io
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------- DATABASE SETUP ----------------
DATABASE_URL = "postgresql+psycopg2://postgres@localhost:5432/logs"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    role = Column(Text)
    content = Column(Text)


Base.metadata.create_all(bind=engine)

# ---------------- FASTAPI APP ----------------
app = FastAPI()

# Allow frontend (Next.js) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store logs in memory (for the current session)
logs_store = {"default": ""}


# ---------------- ENDPOINTS ----------------
@app.post("/upload")
async def upload_log(file: UploadFile):
    contents = await file.read()
    text_data = contents.decode("utf-8", errors="ignore")
    logs_store["default"] = text_data
    return {"status": "ok", "lines": len(text_data.splitlines())}


@app.post("/chat")
async def chat(message: str = Form(...)):
    if not logs_store.get("default"):
        return {"answer": "Please upload a log file first."}

    logs = logs_store["default"]

    system_prompt = """
    You are a log analysis assistant.
    You will be given raw server logs and a user question.
    Your job is to:
    1. Identify and extract errors and warnings.
    2. Suggest potential fixes.
    3. Respond in clear text (Markdown allowed).
    """

    # Save user message
    with SessionLocal() as db:
        db.add(ChatHistory(role="user", content=message))
        db.commit()

    # Stream responsee
    def stream():
        buffer = ""
        for chunk in ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Logs:\n{logs}\n\nUser request: {message}"},
            ],
            stream=True,
        ):
            token = chunk["message"]["content"]
            buffer += token
            yield token

        # Save assistant reply after streaming endss
        with SessionLocal() as db:
            db.add(ChatHistory(role="assistant", content=buffer))
            db.commit()

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/history")
def get_history():
    with SessionLocal() as db:
        records = db.query(ChatHistory).all()
        return [{"role": r.role, "content": r.content} for r in records]
