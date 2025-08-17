from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain.schema import HumanMessage

from app import utils
from app.models import RetrieverState, OrganizerState
from app.nodes import (
    retriever_node,
    organizer_node,
    podcaster_node,
    RESEARCH_TOOLS_NODE,
)

# LangGraph setup 
graph_builder = StateGraph(RetrieverState)
graph_builder.add_node("retriever", retriever_node)
graph_builder.add_node("tools", RESEARCH_TOOLS_NODE)
graph_builder.add_conditional_edges("retriever", tools_condition)
graph_builder.add_edge(START, "retriever")
graph_builder.add_edge("tools", "retriever")
subgraph = graph_builder.compile()

graph_builder2 = StateGraph(OrganizerState)
graph_builder2.add_node("organizer", organizer_node)
graph_builder2.add_node("podcaster", podcaster_node)
graph_builder2.add_edge("organizer", "podcaster")
graph_builder2.add_node("retriever_subgraph", subgraph)
graph_builder2.add_edge("retriever_subgraph", "organizer")
graph_builder2.set_entry_point("retriever_subgraph")
graph = graph_builder2.compile()

# FastAPI 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure ./output exists and serve it at /files
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(OUTPUT_DIR)), name="files")


class Prompt(BaseModel):
    topic: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate_transcript")
def generate_transcript(prompt: Prompt):
    topic = (prompt.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must not be empty.")

    result = graph.invoke({"messages": [HumanMessage(content=topic)]})
    transcript = result.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=500, detail="Failed to generate transcript.")

    # Save transcript under output/
    name = f"{uuid.uuid4().hex}.txt"
    path = OUTPUT_DIR / name
    path.write_text(transcript, encoding="utf-8")

    # Return SAME-ORIGIN RELATIVE URL
    transcript_url = f"/files/{name}"

    return {
        "topic": result.get("topic", topic),
        "cleaned_content": result.get("cleaned_content", ""),
        "transcript": transcript,          # inline preview
        "transcript_url": transcript_url,  # relative URL
        "filename": name,
        "mime_type": "text/plain; charset=utf-8",
    }


@app.post("/generate_podcast")
def generate_podcast(prompt: Prompt):
    topic = (prompt.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic must not be empty.")

    result = graph.invoke({"messages": [HumanMessage(content=topic)]})
    transcript = result.get("transcript")
    if not transcript:
        raise HTTPException(status_code=500, detail="Failed to generate transcript.")

    # TTS should write into output/ and return that path
    try:
        saved_path = Path(utils.tts(transcript))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    # If utils.tts saved elsewhere, move it into output/
    if saved_path.parent.resolve() != OUTPUT_DIR.resolve():
        target = OUTPUT_DIR / saved_path.name
        saved_path.replace(target)
        saved_path = target

    # Return SAME-ORIGIN RELATIVE URL
    audio_url = f"/files/{saved_path.name}"

    return {
        "topic": result.get("topic", topic),
        "cleaned_content": result.get("cleaned_content", ""),
        "transcript": transcript,
        "podcast": audio_url,          
        "filename": saved_path.name,
        "mime_type": "audio/wav",
        "local_path": str(saved_path),
    }
