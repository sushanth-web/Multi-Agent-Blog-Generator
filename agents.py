import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def get_llm(model_name:str = "openai/gpt-oss-20b", temperature : float = 0.5):
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model = model_name, temperature = temperature, api_key = api_key)
    return llm

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    {"role":"system", "content":"""
        "You are a Research agent, Given a blog topic and target audience ,produce a clear, structured research outline."
        "1. 5-7 key points the blog should cover \n"
        "2. Important facts, stats, or examples for each point \n"
        "3. suggested angle or hook\n"
        "Be concise. Use bullet points. Do Not write the full blog yet."


    """},
    {"role":"user","content":"""Topic:{topic}, Audience:{audience}, {revision_hints}, Write a Research outline now."""}
])

def researcher_agent(llm: ChatGroq, topic:str, audience:str, feedback:str = "") -> str:
    revision_hints = f"The human provided this feedback on your previous research - please adress it : {feedback}"
    if not feedback:
        revision_hints = "This is your first attempt"

    chain = RESEARCH_PROMPT | llm
    result = chain.invoke({
        "topic":topic,
        "audience":audience,
        "revision_hints":revision_hints
    })
    return result.content


WRITER_PROMPT = ChatPromptTemplate.from_messages([
    {"role":"system", "content":"""
        "You are a Blog writer agent, using the research notes provided write a complete engaging blog"
        "Rules:\n"
        "1.Length: 500 - 800 words"
        "2.Structure: catchy title, intro hook, 3-5 sections with H2 headings, conclusion \n"
        "3. Tone : clear, friendly, suited to the target audience\n"
        "4. Use markdown formatting
        "Do Not add a 'wordcount' line at the end."


    """},
    {"role":"user","content":"""Topic:{topic}, Audience:{audience},Research Notes:{research}, {revision_hints}, Write the full blog post now."""}
])

def writer_agent(llm: ChatGroq, topic:str, audience:str,research:str, feedback:str = "") -> str:
    revision_hints = f"The human provided this feedback on your previously written blog and asked for these changes - please address them : {feedback}, please apply these while writing the blog"
    if not feedback:
        revision_hints = "This is your first attempt"

    chain = WRITER_PROMPT | llm
    result = chain.invoke({
        "topic":topic,
        "audience":audience,
        "research":research,
        "revision_hints":revision_hints
    })
    return result.content

EDITOR_PROMPT = ChatPromptTemplate.from_messages ([
    {"role":"system", "content":"""
        "You are an Editor Agent the final quality gate before publishing.\n"
        "Take the draft and produce the FINAL polished version. Specifically:\n"
        "- Fix grammar, spelling, and awkward phrasing\n"
        "- Tighten wordy sentences\n"
        "- Improve flow and transitions between sections\n"
        "- Make the title and intro more compelling if need\n"
        "Keep the same structure and markdown formatting\n"
        "Output only the final polished blog post no commentary."
    """},
    { "role":"user", "content":"""
        Topic:{topic},
        draft:{draft}

        Return the published blog post
    """}
])

def editor_agent(llm: ChatGroq, topic:str, draft:str) -> str:
    chain = EDITOR_PROMPT | llm
    result = chain.invoke({
        "topic":topic,
        "draft":draft
    })
    return result.content