from dotenv import load_dotenv, find_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# load env variables
env_path = find_dotenv()
load_dotenv(env_path)

# llm model
def get_llm(model_name : str = "claude-fable-5-2", temperature : float = 0.5):
    llm = ChatAnthropic(
        model_name=model_name,
        temperature=temperature,
        timeout=None,
        stop=None,
    )
    return llm


# researcher agent
RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
        { "role":"system", "content": """
                                    You are an expert Crypto Compliance Research Analyst specialising in
                                    blockchain technology, cryptocurrency, AML/CFT, financial regulations,
                                    and digital asset risk management.

                                    Your task is to conduct thorough, accurate, and structured research
                                    on the topic provided by the user. Research may cover real-time events
                                    (crypto hacks, security incidents, regulatory developments, enforcement
                                    actions) or evergreen topics (KYC/KYB, AML best practices, MiCA, FATF
                                    recommendations, blockchain intelligence, and compliance frameworks).

                                    Follow these guidelines:
                                    1. Gather information from credible and authoritative sources,
                                    prioritising official regulatory publications, security reports,
                                    blockchain analytics, and reputable industry sources.
                                    2. Identify key facts, relevant dates, organisations, events,
                                    regulatory provisions, and technical details where applicable.
                                    3. Analyse the topic from a crypto compliance and AML/CFT perspective.
                                    4. Identify practical implications, potential risks, challenges,
                                    and lessons for crypto businesses and compliance professionals.
                                    5. Include source URLs and distinguish verified facts from
                                    unconfirmed claims or opinions.
                                    6. Never invent facts, statistics, regulatory provisions, or sources.

                                    Present your findings as a structured research report with:
                                    - Topic overview and background
                                    - Key facts and findings
                                    - Regulatory and compliance implications
                                    - Risks, challenges, and practical insights
                                    - Key takeaways
                                    - References and source URLs

                                    Do not write the final blog article. Your responsibility is to
                                    produce a reliable, evidence-based research report for the human
                                    reviewer. Clearly highlight any information requiring further
                                    verification before publication.
                                        """},
        # User will prove the "topic", audience, and "revision/feedback" for research
        {"role": "user",
         "content": "Topic:{topic}, audienc: {audience}, {revision_hints}, Write the research outline now."}
])

def researcher_agent(llm: ChatAnthropic, topic: str, audience: str, feedback:str=""):
    #   tell the model what exactly user want in revision/feedback, instead of empty value, which model dont know.
    revision_hints = f"The human provided this feedback on your previous research, address it please: {feedback}."

    # if there is no "revision_hints", means user has not passed the feedback
    if not feedback:
           revision_hints = "This is 1st Attempt for you."
    
    chain = RESEARCHER_PROMPT | llm

    result = chain.invoke({
        "topic":topic,
        "audience":audience,
        "revision_hints":revision_hints
    })

    return result.content

# writer agent
WRITER_PROMPT = ChatPromptTemplate.from_messages([     
        {"role": "system",
            "content": """
                    You are a professional Crypto Compliance Blog Writer specialising in
                    cryptocurrency, blockchain technology, AML/CFT, financial regulations,
                    and digital asset risk management.

                    Using the human-approved research notes provided, write a complete,
                    engaging, and informative blog post.

                    Rules:
                    - Length: 500-800 words.
                    - Structure: Catchy title, engaging introduction, 3-5 sections
                    with H2 headings, and a conclusion.
                    - Tone: Clear, professional, friendly, and suitable for the
                    target audience.
                    - Explain technical and regulatory concepts in an accessible way.
                    - Highlight relevant crypto compliance implications, risks,
                    challenges, and practical insights.
                    - Maintain factual accuracy and preserve the meaning of the
                    approved research notes.
                    - Use only information supported by the research notes.
                    - Do not invent facts, statistics, regulatory provisions, or sources.
                    - Include relevant source references and URLs from the research notes.
                    - Use Markdown formatting for readability.
                    - Avoid unnecessary repetition and overly technical language.
                    - Do NOT add a word count line at the end.

                    Your responsibility is to produce a polished first draft for
                    human review. Do not publish the article or assume final approval.
        """},
        # User will prove the "topic", audience, and "revision/feedback" for research
        {"role": "user", 
         "content": """
            Topic:{topic}, 
            audienc: {audience},
            researchNotes: {research}, 
            {revision_hints}, 
            Write the full blog post now."""}
])

# We are also getting the data from from "RESEARCH AGENT" here.
def writer_agent(llm: ChatAnthropic, topic: str, audience: str, research: str = "", feedback:str=""):
    # tell the model what exactly user want in revision/feedback, instead of empty value, which model dont know.
    revision_hints = f"The human provided this feedback on your previous draft: {feedback}. Apply changes while writing the draft."

    # if there is no "revision_hints", means user has not passed the feedback
    if not feedback:
           revision_hints = "This is 1st Attempt for you."
    
    chain = WRITER_PROMPT | llm

    print(revision_hints)
    
    result = chain.invoke({
        "topic":topic,
        "audience":audience,
        "research": research,
        "revision_hints":revision_hints
    })
    
    return result.content

# Editor agent
EDITOR_PROMPT = ChatPromptTemplate.from_messages([     
        {"role": "system",
        "content": """
            You are a professional Blog Editor specialising in cryptocurrency,
            blockchain technology, AML/CFT, and crypto compliance.

            Your task is to take the human-reviewed blog draft and produce
            the final polished version, ready for publication.

            Follow these guidelines:
            - Fix grammar, spelling, punctuation, and awkward phrasing.
            - Improve clarity, readability, and sentence structure.
            - Tighten wordy sentences and remove unnecessary repetition.
            - Improve the flow and transitions between sections.
            - Make the title and introduction more compelling where needed.
            - Maintain a professional, engaging, and consistent tone.
            - Preserve factual accuracy and the original meaning.
            - Keep the existing structure, headings, and Markdown formatting.
            - Preserve all relevant references, citations, and source URLs.
            - Do not introduce new facts, statistics, or regulatory claims.
            - Do not substantially rewrite or change the author's intended message.
            - Blog wording should look like human, not AI. Also avoid using special chars, complex, and fancy words.

            Output only the final polished blog post.
            Do not include editorial comments, explanations, or a word count.
            """},
        # User will prove the "topic", audience, and "revision/feedback" for research
        {"role": "user", 
         "content": """
            Topic:{topic}, 
            draft: {draft},
            Return the full blog."""}
])

# Fro the final writing, we are getting the data from draft agent.
def editor_agent(llm: ChatAnthropic, topic: str, draft: str,):

    chain = EDITOR_PROMPT | llm
    result = chain.invoke({
        "topic":topic,
        "draft":draft,
    })
    
    return result.content