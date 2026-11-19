import logging
import re
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from graph import build_blog_graph


logger = logging.getLogger(__name__)


def run_graph(graph, config, graph_input, status):
    with st.spinner(status):
        try:
            graph.invoke(graph_input, config=config)
        except Exception:
            logger.exception("Blog generation failed")
            st.session_state.blog_retry = (graph_input, status)
        else:
            st.session_state.pop("blog_retry", None)
    st.rerun()


def show_start(graph):
    with st.form("blog_inputs"):
        topic = st.text_input("Blog topic", placeholder="AML in cryptocurrencies")
        audience = st.text_input(
            "Target audience", placeholder="Compliance officers and beginners"
        )
        submitted = st.form_submit_button("Generate research", type="primary")

    if submitted:
        topic, audience = topic.strip(), audience.strip()
        if not topic or not audience:
            st.warning("Enter a topic and target audience to continue.")
            return

        config = {"configurable": {"thread_id": uuid4().hex}}
        st.session_state.blog_config = config
        run_graph(
            graph, config, {"topic": topic, "audience": audience}, "Generating research…"
        )


def show_review(graph, config, checkpoint_id, review):
    stage = review["stage"]
    if stage == "researcher_review":
        title, content = "Review research", review["research"]
        approve_label, revise_label = "Approve and write draft", "Revise research"
        approve_status, revise_status = "Writing the draft…", "Revising research…"
    else:
        title, content = "Review draft", review["draft"]
        approve_label, revise_label = "Approve and finalize", "Revise draft"
        approve_status, revise_status = "Finalizing the blog…", "Revising the draft…"

    st.header(title)
    st.markdown(content)
    st.divider()
    with st.form(f"review_{checkpoint_id}", enter_to_submit=False):
        feedback = st.text_area("Revision feedback")
        approve = st.form_submit_button(approve_label, type="primary")
        revise = st.form_submit_button(revise_label)

    if approve:
        decision = Command(resume={"action": "approve", "feedback": ""})
        run_graph(graph, config, decision, approve_status)
    elif revise:
        feedback = feedback.strip()
        if not feedback:
            st.warning("Enter feedback to request a revision.")
            return
        decision = Command(resume={"action": "revise", "feedback": feedback})
        run_graph(graph, config, decision, revise_status)


def show_final(topic, final_blog):
    st.header("Final blog")
    st.markdown(final_blog)
    filename = re.sub(r"[^\w-]+", "-", topic.lower()).strip("-_")[:80] or "blog"
    st.download_button(
        "Download Markdown", data=final_blog, file_name=f"{filename}.md", mime="text/markdown"
    )


def show_retry(graph, config, snapshot, pending_action):
    graph_input, status = pending_action
    st.error("Blog generation stopped. Try the current step again.")
    if st.button("Retry", type="primary"):
        # A failed node can be replayed from its checkpoint after the review was accepted.
        retry_input = None if snapshot.next and not snapshot.interrupts else graph_input
        run_graph(graph, config, retry_input, status)

    draft = snapshot.values.get("draft")
    research = snapshot.values.get("research")
    if draft:
        st.header("Latest draft")
        st.markdown(draft)
    elif research:
        st.header("Latest research")
        st.markdown(research)


def main():
    st.set_page_config(page_title="Blog Generator", page_icon="📝")
    st.title("Crypto Compliance Blog Assistant")
    st.write("A complete AI-assisted (Human in the Loop) workspace for researching, " \
    "reviewing, drafting, editing, and publishing cryptocurrency compliance articles.")

    if "blog_graph" not in st.session_state:
        st.session_state.blog_graph = build_blog_graph()

    graph = st.session_state.blog_graph
    config = st.session_state.get("blog_config")
    if config is None:
        show_start(graph)
        return

    snapshot = graph.get_state(config)
    values = snapshot.values
    if values.get("topic"):
        st.text(f"Topic: {values['topic']}")
        st.text(f"Audience: {values['audience']}")

    if st.button("Start a new blog"):
        st.session_state.pop("blog_config")
        st.session_state.pop("blog_retry", None)
        st.rerun()

    pending_action = st.session_state.get("blog_retry")
    if pending_action is not None:
        show_retry(graph, config, snapshot, pending_action)
    elif snapshot.interrupts:
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        show_review(graph, config, checkpoint_id, snapshot.interrupts[0].value)
    elif values.get("final_blog"):
        show_final(values["topic"], values["final_blog"])
    else:
        st.error("The workflow ended without a final blog. Start a new blog to try again.")


if __name__ == "__main__":
    main()
