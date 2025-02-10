import streamlit as st
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from linkedin_scraper import ScrapinAPI, get_profile_data, get_Job_data
from langchain_core.messages import AIMessage, HumanMessage
from astream_events_handler import invoke_our_graph   # Utility function to handle events from astream_events from graph
load_dotenv()

# app config
st.set_page_config(page_title="LearnTube CareerSync", page_icon="🤖")
st.title(":rainbow[_LearnTube_] :blue[CareerSync]")
st.markdown("*Your AI-Powered LinkedIn and Career Strategist!*")

# Functions to validate respective URL
def is_valid_linkedin_profile_url(url):
    if not url.startswith("https://www.linkedin.com/in/"):
        return False
    return True

def is_valid_linkedin_job__url(url):
    if not url.startswith("https://www.linkedin.com/jobs/"):
        return False
    return True

# Initialize session state
# if "isProfile" not in st.session_state:
#     st.session_state.isProfile = False
# if "isjob" not in st.session_state:
#     st.session_state.isjob = False
if "messages" not in st.session_state:
    # Default initial message to render in message state
    st.session_state["messages"] = [AIMessage(content="Hi! 👋 I’m your AI career assistant. I can optimize your LinkedIn profile, analyze job fit, generate personalized cover letters, career recommendations and networking opportunities & events. Share your profile or a job link to get started!")]


# Disable chat input until response is received.
# if "chat_input" not in st.session_state:
#     st.session_state.chat_input = False
# def disable_callback():
#     st.session_state.chat_input = True
# disabled=st.session_state.chat_input, on_submit=disable_callback

# Sidebar
# with st.sidebar:
#     st.header("Settings")
#     profile_url = st.text_input("Linkedin Profile URL")

#     # Validate URL and show error message if invalid
#     if profile_url:
#         if not is_valid_linkedin_profile_url(profile_url):
#             st.error("Please enter a valid LinkedIn Profile URL", icon="🚨")
#         else:
#             # Call get_data function
#             profile_data = get_profile_data(profile_url)
#             if profile_data.get("success", False):
#                 st.session_state.isProfile = True
#                 st.session_state.profile_data = profile_data
#             else:
#                 st.error(profile_data.get("msg", "An unknown error occurred") + " Please check the URL and try again", icon="🚨")

#     job_url = st.text_input("Linkedin Job URL")
#     if job_url:
#         if not is_valid_linkedin_job__url(job_url):
#             st.error("Please enter a valid LinkedIn Job URL", icon="🚨")
#         else:
#             # Call get_data function
#             job_data = get_Job_data(job_url)
#             if job_data.get("success", False):
#                 st.session_state.isjob = True
#                 st.session_state.job_data = job_data
#             else:
#                 st.error(job_data.get("msg", "An unknown error occurred") + " Please check the URL and try again", icon="🚨")

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we are storing them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# Takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handles graph events using the callback mechanism
    with st.chat_message("assistant"):
        # create a placeholder container for streaming and any other events to visually render here
        placeholder = st.container()
        response = st.write(asyncio.run(invoke_our_graph(st.session_state.messages, placeholder)))
        # print("-----------finala-----response--------------------------------",response)
        st.session_state.messages.append(AIMessage(str(response)))
        # print("\nst.session_state.messages---------\n",st.session_state.messages)
        

