from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import Literal
# Import state
from state import BlogState

# import agents and llm
from agent import get_llm, researcher_agent, writer_agent, editor_agent


MAX_RIVISIONS : int = 3

# NODES
# 1. research node
def researcher_node(state : BlogState):
    """Research agent generate (or revise) the research outline."""
    # check if the user has provided feedback for revision
    if state.research_feedback:
        feedback = state.research_feedback
    else:
        feedback = ""

    research_outline = researcher_agent(get_llm(), state.topic, state.audience, feedback)
    state.research = research_outline

    return state

# 2. Human Interuption
def human_review_research_node(state : BlogState):
    # """Human review the research outline and provide feedback for revision."""
    # # check if the user has provided feedback for revision
    # if state.research_feedback:
    #     feedback = state.research_feedback
    # else:
    #     feedback = ""

    # # if the user has provided feedback, check if the revision count is less than MAX_RIVISIONS
    # if feedback and state.revision_count < MAX_RIVISIONS:
    #     state.revision_count += 1
    #     return interrupt("Research outline requires revision. Please provide feedback for revision.")
    # elif feedback and state.revision_count >= MAX_RIVISIONS:
    #     return interrupt("Maximum revisions reached. Please proceed to the next step.")
    # else:
    #     return interrupt("Research outline is ready for writing. Please proceed to the next step.")
    """Pause and ask human to approve the research outline, or give feedback."""
    decision = interrupt({
        "stage":"researcher_review",
        "research":state.research,
        "instruction": (
            "Reply with 'approve' to continue writing",
            "or provide feedback for revision, and send it back to researcher."
        )
    })

    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approved", "approve", "ok", "yes", "proceed", ""] else "revise"
        feedback = "" if action == "approve" else text

    state.research_feedback = feedback
    return state

# 2. writer node
def writer_node(state: BlogState):
    """Writer agent generate (or revise) the blog draft."""
    llm = get_llm()
    draft_data = writer_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        research=state.research,
        feedback=state.draft_feedback,
    )
    state.draft = draft_data
    state.draft_feedback = ""

    return state

# 3. writer review node
def human_review_draft_node(state : BlogState):

    """Pause and ask human to approve the drqft outline, or give feedback."""
    decision = interrupt({
        "stage":"draft_review",
        "draft":state.draft,
        "instruction": (
            "Reply with 'approve' to continue editor.",
            "or provide feedback for revision, and send it back to draft writer."
        )
    })

    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ["approved", "approve", "ok", "yes", "proceed", ""] else "revise"
        feedback = "" if action == "approve" else text

    state.draft_feedback = feedback
    return state

# 2. editor node
def editor_node(state:BlogState):
    llm = get_llm()
    final = editor_agent(
        llm=llm,
        topic=state.topic,
        draft=state.draft
    )

    state.final_blog = final
    return state

# ---------------------------
# Conditional Nodes
# ---------------------------

# 1. After Research
def route_after_research_review(state:BlogState) -> Literal["research", "writer"]:
    """Route to research or draft based on human feedback."""
    if state.research_feedback:
        return "research"
    else:
        return "writer"

# 2. After Draft
def route_after_draft_review(state:BlogState) -> Literal["writer", "edit"]:
    """Route to draft/writer or edit based on human feedback."""
    if state.draft_feedback and state.revision_count < MAX_RIVISIONS:
        return "writer"
    else:
        return "edit"

# ---------------------------
# Building Graph
# ---------------------------
def build_blog_graph():
    builder = StateGraph(BlogState)

    # ading nodes
    builder.add_node("research", researcher_node)
    builder.add_node("review_research", human_review_research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review_draft", human_review_draft_node)
    builder.add_node("edit", editor_node)

    # adding edges
    builder.add_edge(START, "research")
    builder.add_edge("research", "review_research")
    builder.add_conditional_edges("review_research", 
                                  route_after_research_review,
                                  {"research": "research", "writer":"writer"}) #route return output decide, either move to "research" OR "writer"
    builder.add_edge("writer", "review_draft")
    builder.add_conditional_edges("review_draft", 
                                  route_after_draft_review,
                                  {"writer":"writer", "edit":"edit"}) # route return output decide, either move to "writer" OR "edit"
    builder.add_edge("edit", END)

    GRAPH = builder.compile(checkpointer=InMemorySaver())
    return GRAPH