
import os
from langchain_core.tools import tool, StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain.agents import AgentExecutor, create_react_agent
# from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from agents_prompts import LINKEDIN_PROFILE_ANALYST_AGENT, REACT_AGENT_TEMPLATE
import agents_prompts
from tools import process_linkedin_data
from sample_data import data
from langchain import hub
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

is_profile = False
# function to load Gemini Pro model and get repsonses
# gemini-pro, gemini-2.0-flash-exp, gemini-1.5-pro, gemini-2.0-flash, gemini-1.5-flash, gemini-2.0-flash-thinking-exp-01-21
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=1,
    # streaming=True,
)

# gpt-4o    o1-mini
# llm = ChatOpenAI(model="gpt-4o")

# search_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
web_searcher_tool = TavilySearchResults(max_results=5)
@tool
def get_linkedin_profile_data():
    """Fetch stored LinkedIn profile data."""
    if is_profile:
        photo, background, cleaned = process_linkedin_data(data)
        print("----------------------cleaned----------------\n", cleaned)
        print("----------------------cleaned----------------\n")
        return cleaned
    else:
        print("-----------------No LinkedIn profile URL provided------------")
        return  {
            "error": "No LinkedIn profile URL provided",
            "action": "End the process and prompt the user with: 'Please provide your valid LinkedIn profile URL for analysis and try again."
        }

# {"error": "No LinkedIn profile URL provided. Please enter a valid URL and try again."}

toolss = [get_linkedin_profile_data]
tools = [get_linkedin_profile_data]

# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate

system_message = LINKEDIN_PROFILE_ANALYST_AGENT

# agent = create_tool_calling_agent(llm, tools, system_message)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# ans = agent_executor.invoke({"messages": [("user", "Can you help me optimize my linkedin profile")]})
# print("-------------------------ans----\n",ans['output'])
# system_message = "You are a helpful assistant. Respond only in Spanish."
# # This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

langgraph_agent_executor = create_react_agent(llm, tools, prompt=LINKEDIN_PROFILE_ANALYST_AGENT)

messages = langgraph_agent_executor.invoke({"messages": [("user", "Can you help me optimize my linkedin profile")]})
print("messages----\n",messages)
print("---fina;l----messages----\n",messages["messages"][-1].content)
# llm_with_tools = llm.bind_tools([get_linkedin_profile_data])


# ----------------------------------------------------------
# LINKEDIN_PROFILE_ANALYST_AGENT
# "You are linkedin profile analyzer address the user query and call get_linkedin_profile_data to get user data"
# research_agent = create_react_agent(
#     llm, tools=[get_linkedin_profile_data], prompt=LINKEDIN_PROFILE_ANALYST_AGENT
# )
# inputs = {"messages": "User Input: can you Optimize my linkedin profile"}
# # inputs = {"messages": [("user", "What's your name? And what's the weather in SF?")]}
# result = research_agent.invoke(inputs)
# print("final ans", result["messages"][-1].content)


# prompt_template = PromptTemplate.from_template("You are linkedin profile analyzer address the user query and call get_linkedin_profile_data to get user data \nUser Query:{input}")
# prompt_template = hub.pull("hwchase17/react")
# print(LINKEDIN_PROFILE_ANALYST_AGENT)
test_prompt = PromptTemplate.from_template(REACT_AGENT_TEMPLATE)
# test_prompt = PromptTemplate.from_template(LINKEDIN_PROFILE_ANALYST_AGENT)

# agent = create_react_agent(llm=llm, tools=toolss, prompt="You are linkedin profile analyzer address the user query and call get_linkedin_profile_data to get user data \nUser Query:{input}")

# agent_executor = AgentExecutor(agent=agent, tools=toolss, verbose=True)

# res = agent_executor.invoke({"input": "Optmize my linkedin profile"})
# print("final ans",res)

# Make sure you have defined 'llm' before this line
# agent = create_react_agent(llm=llm, tools=toolss, prompt=test_prompt)

# agent_executor = AgentExecutor(agent=agent, tools=toolss, verbose=True, handle_parsing_errors=True)

# res = agent_executor.invoke({"input": "Can you help me optimize my linkedin profile"})
# print("final ans------\n", res)
# # print("----------------------------------------------------------------------------------------")
# # print("final ans--with oyutput----\n", res['output'])


# prompt = hub.pull("hwchase17/react")
# print(prompt.template)
# print("----------------------------------------------------------------")
# prompt = hub.pull("hwchase17/react-chat")
# print(prompt.template)



# ========================================================================================================
# from agents_prompts import LINKEDIN_PROFILE_ANALYST_AGENT
# import agents_prompts
# from tools import process_linkedin_data
# from tempData import data
# import dspy
# from dspy.predict.react import Tool


# # # gemini-pro, gemini-2.0-flash-exp, gemini-1.5-pro, gemini-2.0-flash, gemini-1.5-flash
# # lm = dspy.LM('gemini/gemini-1.5-pro', api_key='AIzaSyABGjbDfPTNue27tyioNAn5oHyvIVKKw5U')
# # o1-mini  gpt-4o
# lm = dspy.LM('openai/gpt-4o', api_key='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')
# dspy.configure(lm=lm)




# def get_linkedin_profile_data():
#     """Fetch stored LinkedIn profile data."""
#     # if st.session_state.isProfile:
#     #     profile_data = st.session_state.get('profile_data', 'Please enter a valid URL and try again.')
#     #     return profile_data
#     photo, background, cleaned = process_linkedin_data(data)
#     print("----------------------cleaned----------------\n", cleaned)
#     print("----------------------cleaned----------------\n")
#     return cleaned
#     # return {"error": "No LinkedIn profile URL provided. Please enter a valid URL and try again."}

# instructions = LINKEDIN_PROFILE_ANALYST_AGENT
# signature = dspy.Signature("Input_Data -> answer: str", instructions)
# react = dspy.ReAct(signature, tools=[Tool(get_linkedin_profile_data)])


# inputs = {
#     "user_query_details": "can you help me optmize my linkedin profile",
#     "worker_outputs": "No output yet"
# }

# response  = react(Input_Data=agents_prompts.REACT_INVOKE_TEMPLATE.format(**inputs))

# # pred = react(Input_data="can you optmize my linkedin profile")
# print("------final ans----\n",response.answer)



