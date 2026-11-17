from pydantic import BaseModel

# Difining the STATE

class BlogState(BaseModel):
    # User input
    topic: str = ""
    audience: str = ""

    # Research
    research : str = ""
    research_feedback : str = ""

    # Writer
    draft : str = ""
    draft_feedback : str = ""

    # Editor
    final_blog : str = ""

    # Metadata
    revision_count : int = 0

