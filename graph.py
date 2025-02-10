import os
from typing import Annotated, Literal, List, Optional
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.types import Command
from langgraph.graph import MessagesState, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel, Field
from data_process import format_chat_data, process_job_data, process_linkedin_data
from sample_data import data
import agents_prompts
import streamlit as st
import linkedin_scraper
from dotenv import load_dotenv
load_dotenv()


# LLM
# gemini-2.0-flash-exp, gemini-1.5-pro, gemini-2.0-flash, gemini-1.5-flash, gemini-2.0-flash-thinking-exp-01-21
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=1,
    # streaming=True,
)

# Other Models
# llm = ChatOpenAI(model="o1-mini")
# llm = ChatAnthropic(model="Claude 3.5 Haiku", api_key=os.getenv("ANTHROPIC_API_KEY"))

#Web Search Tool
web_searcher_tool = TavilySearchResults(max_results=5)

@tool
def get_linkedin_profile_data(linkedin_url: str) -> dict:
    """
    Extract person's profile data from LinkedIn URL.
    Args:
        linkedin_url (str): Valid LinkedIn profile URL in format:
                            https://www.linkedin.com/in/xxxxxxxxxxxxx/
    Returns:
        dict: Response JSON containing profile data
    """
    # Validate URL input
    if not linkedin_url:
        return {"success": False, "title": "Invalid Input", "msg": "No LinkedIn Profile URL provided." }
    try:
        profile_data = linkedin_scraper.get_profile_data(linkedin_url)
        if profile_data['success'] == False:
            return profile_data
        photo, background, formatted_data = process_linkedin_data(profile_data)
        return formatted_data
    except Exception as e:
            return {"success": False, "title": "Error", "msg": f"Error retrieving Linkedin Profile data: {str(e)}"}

@tool
def get_job_description_data(linkedin_job_url: str) -> dict:
    """
    Extract Job data from LinkedIn Job URL.
    Args:
        linkedin_job_url (str): Valid LinkedIn URL of the job with the following formats: 
        https://www.linkedin.com/jobs/collections/recommended/?currentJobId=XXXXXXXXXX, 
        https://www.linkedin.com/jobs/search/?currentJobId=XXXXXXXXXX or 
        https://www.linkedin.com/jobs/view/XXXXXXXXXX/.
    Returns:
        dict: Response JSON containing job data
    """
    if not linkedin_job_url:
        return {"success": False, "title": "Invalid Input", "msg": "No LinkedIn Job URL provided." }
    try:
        profile_data = linkedin_scraper.get_Job_data(linkedin_job_url)
        if profile_data['success'] == False:
            return profile_data
        formatted_data = process_job_data(profile_data)
        return formatted_data
    except Exception as e:
            return {"success": False, "title": "Error", "msg": f"Error retrieving job description data: {str(e)}"}
    

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    next: str
    output: str
    # Custom keys for additional data can be added here such as - conversation_id: str

class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["general_chat_handler", "linkedin_profile_analyst", "job_fit_analyst", "career_advisor", "cover_letter_generator", "opportunity_tracker", 'FINISH']

class GeneralChatAgent(BaseModel):
    reponse: str

# Function to decide whether to continue tool usage or end the process
def supervisor_node(state: GraphsState) -> Command[Literal["general_chat_handler", "linkedin_profile_analyst", "__end__"]]:
    superviso_data = {
        "supervisor_prompt": agents_prompts. SUPERVISOR_PROMPT,
        "user_query_details": state['user_query'],
        "worker_output": state['output']

    }
    response = llm.with_structured_output(Router).invoke(agents_prompts.FINAL_SUPERVISOR_PROMPT.format(**superviso_data))
    print(type(response))
    goto = response.next
    final_resp = ""
    if goto == "FINISH":
        inputs = {
            "user_query_details": state['user_query'],
            "worker_outputs": state['output']
        }
        final_resp = llm.invoke(agents_prompts.FINAL_OUTPUT_FORMATTER.format(**inputs))
        # goto = END
        return Command(goto=END, update={"next": goto,
                                      "messages": [ AIMessage(content=final_resp.content, name="supervisor")] })
    
    return Command(goto=goto, update={"next": goto,})


def general_chat_handler(state: GraphsState) -> Command[Literal["supervisor"]]:
    inputs = {
        "user_query_details": state['user_query'],
        "worker_outputs": state['output']
    }
    response = llm.with_structured_output(GeneralChatAgent).invoke(agents_prompts.GENERAL_CHAT_AGENT.format(**inputs))
    if state['output'] == "No outputs yet":
        state['output'] = ""
    formatted_output = state["output"] + "\nWorker-general_chat_handler: \n" + response.reponse
    return Command(
        update={
            "output": formatted_output
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

# *----------------------linkedin_profile_analyst-----------------------------
def linkedin_profile_analyst(state: GraphsState) -> Command[Literal["supervisor"]]:
    inputs = {
        "user_query_details": state['user_query'],
        "worker_outputs": state['output']
    }
    query = agents_prompts.REACT_INVOKE_TEMPLATE.format(**inputs)
    main_prompt = PromptTemplate.from_template(agents_prompts.LINKEDIN_PROFILE_ANALYST_AGENT)

    agent = create_react_agent(llm=llm, tools=[get_linkedin_profile_data], prompt=main_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[get_linkedin_profile_data], verbose=True,  handle_parsing_errors=True, max_iterations=5)

    response = agent_executor.invoke({"input": query})

    if state['output'] == "No outputs yet":
        state['output'] = ""
    formatted_output = state["output"] + "\n" + "Worker-linkedin_profile_analyst:\n" + response['output']

    return Command(
        update={
            "output": formatted_output
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

# *---------------------------job_fit_analyst--------------------------------
def job_fit_analyst(state: GraphsState) -> Command[Literal["supervisor"]]:
    inputs = {
        "user_query_details": state['user_query'],
        "worker_outputs": state['output']
    }
    query = agents_prompts.REACT_INVOKE_TEMPLATE.format(**inputs)
    main_prompt = PromptTemplate.from_template(agents_prompts.JOB_FIT_ABALYSIS_PROMPT)

    agent = create_react_agent(llm=llm, tools=[get_linkedin_profile_data, get_job_description_data], prompt=main_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[get_linkedin_profile_data, get_job_description_data], verbose=True, handle_parsing_errors=True, max_iterations=5)

    response = agent_executor.invoke({"input": query})
    
    if state['output'] == "No outputs yet":
        state['output'] = ""
    formatted_output = state["output"] + "\n" + "Worker-job_fit_analyst:\n" + response['output']

    return Command(
        update={
            "output": formatted_output
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

#* --------------------------career_advisor-----------------------------------------------
def career_advisor(state: GraphsState) -> Command[Literal["supervisor"]]:
    inputs = {
        "user_query_details": state['user_query'],
        "worker_outputs": state['output']
    }
    query = agents_prompts.REACT_INVOKE_TEMPLATE.format(**inputs)
    main_prompt = PromptTemplate.from_template(agents_prompts.CAREER_ADVISOR_PROMPT)

    agent = create_react_agent(llm=llm, tools=[get_linkedin_profile_data, web_searcher_tool], prompt=main_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[get_linkedin_profile_data, web_searcher_tool], verbose=True, handle_parsing_errors=True, max_iterations=5)

    response = agent_executor.invoke({"input": query})    
    if state['output'] == "No outputs yet":
        state['output'] = ""
    formatted_output = state["output"] + "\n" + "Worker-career_advisor:\n" + response['output']

    return Command(
        update={
            "output": formatted_output
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

#* ---------------------------cover_letter_generator----------------------------------
def cover_letter_generator(state: GraphsState) -> Command[Literal["supervisor"]]:
    inputs = {
        "user_query_details": state['user_query'],
        "worker_outputs": state['output']
    }
    query = agents_prompts.REACT_INVOKE_TEMPLATE.format(**inputs)
    main_prompt = PromptTemplate.from_template(agents_prompts.COVER_LETTER_GENERATOR_PROMPT)

    agent = create_react_agent(llm=llm, tools=[get_linkedin_profile_data, get_job_description_data], prompt=main_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[get_linkedin_profile_data, get_job_description_data], verbose=True, handle_parsing_errors=True, max_iterations=5)

    response = agent_executor.invoke({"input": query})
    
    if state['output'] == "No outputs yet":
        state['output'] = ""
    formatted_output = state["output"] + "\n" + "Worker-cover_letter_generator:\n" + response['output']

    return Command(
        update={
            "output": formatted_output
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )

# ----------------------------------------------------opportunity_tracker--------------------------------------------------------
def opportunity_tracker(state: GraphsState) -> Command[Literal["supervisor"]]:
    inputs = {
        "user_query_details": state['user_query'],
        "worker_outputs": state['output']
    }
    query = agents_prompts.REACT_INVOKE_TEMPLATE.format(**inputs)
    main_prompt = PromptTemplate.from_template(agents_prompts.OPPORTUNITY_TRACKER_PROMPT)

    agent = create_react_agent(llm=llm, tools=[web_searcher_tool, get_linkedin_profile_data], prompt=main_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=[web_searcher_tool, get_linkedin_profile_data], verbose=True, handle_parsing_errors=True, max_iterations=5)

    response = agent_executor.invoke({"input": query})
    
    if state['output'] == "No outputs yet":
        state['output'] = ""
    formatted_output = state["output"] + "\n" + "Worker-opportunity_tracker:\n" + response['output']

    return Command(
        update={
            "output": formatted_output
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )



# Define the structure (nodes and directional edges between nodes) of the graph
graph = StateGraph(GraphsState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("general_chat_handler", general_chat_handler)
graph.add_node("linkedin_profile_analyst", linkedin_profile_analyst)

graph.add_node("job_fit_analyst", job_fit_analyst)
graph.add_node("career_advisor", career_advisor)
graph.add_node("cover_letter_generator", cover_letter_generator)
graph.add_node("opportunity_tracker", opportunity_tracker)

graph.add_edge(START, "supervisor")


# Compile the state graph into a runnable object
graph_runnable = graph.compile()

# To generate the graph image
# from IPython.display import Image, display
# display(Image(graph_runnable.get_graph().draw_mermaid_png(output_file_path="graph.png")))


# Without streaming
def invoke_our_graph(st_messages, callables):
    st_messages = format_chat_data(st_messages)
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph_runnable.invoke({"user_query": st_messages, "output": "No outputs yet"}, config={"callbacks": callables})


# test invoke
def test_invoke_our_graph(st_messages):
    return graph_runnable.invoke({"user_query": st_messages, "output": "No outputs yet"})

# Test Query
query = """
Chat History: "No chat history"
Current User Question: "Help me optimize my profile: https://www.linkedin.com/in/niranjan-khedkar123/"
"""

# Help me optimize my profile: https://www.linkedin.com/in/niranjan-khedkar123/

if __name__ == "__main__":
    print("test_response\n",test_invoke_our_graph(query))
