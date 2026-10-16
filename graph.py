from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import Literal

from state import BlogState
from agents import get_llm,researcher_agent, writer_agent, editor_agent

MAX_REVISIONS = 3 

def researcher_node(state:BlogState) -> BlogState:
    """Researcher agent generates or revises the research outline"""
    llm = get_llm()
    research_data = researcher_agent(
        llm = llm,
        topic = state.topic,
        audience = state.audience,
        feedback = state.research_feedback
    )
    state.research = research_data
    state.research_feedback = ""

    return state


def human_review_research_node(state:BlogState):

    """Pause and ask the human to approve the research or send the feedback"""
    decision = interrupt({
        "stage": "researcher review",
        "research":state.research,
        "instructions": (
            "Reply with 'approve to continue to writing",
            "or describe what to change to send it back to the researcher.")
    })
    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "approved", "ok", "yes", "process", ""] else "revise"

        feedback = "" if action == "approve" else text

    state.research_feedback = feedback
    return state


def writer_node(state:BlogState) -> BlogState:
    llm = get_llm()
    writer_data = writer_agent(
            llm = llm,
            topic = state.topic,
            audience = state.audience,
            research = state.research,
            feedback = state.draft_feedback
        )
    state.draft = writer_data
    state.draft_feedback = ""
    
    return state

def human_review_draft_node(state:BlogState):

    """Pause and ask the human to approve the draft or send the feedback"""
    decision = interrupt({
        "stage": "draft review",
        "draft":state.draft,
        "instructions": (
            "Reply with 'approve to continue to writing",
            "or describe what to change to send it back to the researcher.")
    })
    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approve", "approved", "ok", "yes", "process", ""] else "revise"

        feedback = "" if action == "approve" else text

    state.draft_feedback = feedback
    return state

def editor_node(state:BlogState) -> BlogState:
    llm = get_llm()
    final_data = editor_agent(
            llm = llm,
            topic = state.topic,
            draft = state.draft
        )
    state.final_blog = final_data
    
    return state


def route_after_research_review(state:BlogState) -> Literal["research", "writer"]:
    if state.research_feedback:
        return "research"
    else:
        return "writer"

def route_after_draft_review(state:BlogState) -> Literal["writer","edit"]:
    if state.draft_feedback and state.revision_count < MAX_REVISIONS:
        return "writer"
    else:
        return "edit"

def build_blog_graph():
    builder = StateGraph (BlogState)

    #e add nodes
    builder.add_node("research", researcher_node)
    builder.add_node("review_research", human_review_research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review_draft", human_review_draft_node)
    builder.add_node("edit", editor_node)
    
    builder.add_edge(START, "research")
    builder.add_edge("research", "review_research")
    builder.add_conditional_edges("review_research", route_after_research_review, {"research": "research", "writer":"writer"})
    builder.add_edge("writer","review_draft")
    builder.add_conditional_edges("review_draft", route_after_draft_review, {"writer":"writer","edit": "edit"})
    builder.add_edge("edit", END)

    GRAPH = builder.compile(checkpointer=InMemorySaver())
    return GRAPH
    