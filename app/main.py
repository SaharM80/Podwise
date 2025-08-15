from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain.schema import HumanMessage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import utils
from app.models import RetrieverState, OrganizerState
from app.nodes import (
    retriever_node,
    organizer_node,
    podcaster_node,
    RESEARCH_TOOLS_NODE,
)

# Graph Setup 
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

# FastAPI App 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the output folder at /files
app.mount("/files", StaticFiles(directory="output"), name="files")

# Pydantic Model 
class Prompt(BaseModel):
    topic: str

# Endpoints 
@app.post("/generate_transcript")
def transcribe(prompt: Prompt):
    result = graph.invoke({"messages": [HumanMessage(content=prompt.topic)]})
    return {
        "topic": result["topic"],
        "cleaned_content": result["cleaned_content"],
        "transcript": result["transcript"],
    }


@app.post("/generate_podcast")
def podwise(prompt: Prompt):
    # Generate content
    result = graph.invoke({"messages": [HumanMessage(content=prompt.topic)]})
    transcript = result["transcript"]

    # Save audio locally
    saved_path = utils.tts(transcript) 

    # Create a URL for the frontend 
    filename = saved_path.split("/")[-1]  
    audio_url = f"/files/{filename}" 

    return {
        "topic": result["topic"],
        "cleaned_content": result["cleaned_content"],
        "transcript": transcript,
        "podcast": audio_url, 
        "local_path": saved_path  
    }


