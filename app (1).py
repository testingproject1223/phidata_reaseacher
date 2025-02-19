import streamlit as st
import os
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k

# Initialize session state with defaults
session_defaults = {
    "apikey": "",
    "link_count": 10,
    "conversation": [],
    "loading": False,
    "processing": False
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Sidebar configuration
def configure_sidebar():
    st.sidebar.title("Settings")
    
    # API key input
    api_key = st.sidebar.text_input(
        "Google API Key:",
        type="password",
        value=st.session_state.apikey,
        help="Get your API key from https://makersuite.google.com/"
    )
    st.session_state.apikey = api_key
    
    # Link count selector
    st.session_state.link_count = st.sidebar.number_input(
        "Links to analyze:",
        min_value=1,
        max_value=50,
        value=st.session_state.link_count,
        help="Number of articles to process for research"
    )
    
    # Conversation controls
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear Conversation"):
        st.session_state.conversation = []
        st.rerun()

configure_sidebar()

# Validate API key before proceeding
if not st.session_state.apikey:
    st.error("🔑 API key required to continue")
    st.stop()

# Agent initialization with caching
@st.cache_resource(show_spinner=False)
def create_research_agent(api_key: str, link_count: int):
    os.environ["GOOGLE_API_KEY"] = api_key  # Consider using secrets in production
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo(), Newspaper4k()],
        description="You are a senior NYT researcher writing an article on a topic.",
        instructions=[
            f"For a given topic, search for the top {link_count} links.",
            "Then read each URL and extract the article text. If a URL isn't available, ignore it.",
            "Analyze and prepare an NYT-worthy article based on the information."
        ],
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        # Number of historical responses to add to the messages.
        num_history_responses=3,
    )

research_agent = create_research_agent(st.session_state.apikey, st.session_state.link_count)

# Main interface
st.title("📰 NYT Research Assistant")
st.caption("Collaborative AI-powered research platform for investigative journalism")

# Display conversation history
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Handle user input
if user_input := st.chat_input("Enter your research topic or feedback...", disabled=st.session_state.loading):
    if not user_input.strip():
        st.warning("Please enter a valid message")
        st.stop()
    
    st.session_state.conversation.append({"role": "user", "message": user_input})
    st.session_state.loading = True
    st.rerun()

# Process agent response
if st.session_state.loading and not st.session_state.processing:
    st.session_state.processing = True
    try:
        # Use only the latest user query for processing
        latest_query = st.session_state.conversation[-1]["message"]
        with st.spinner("🔍 Conducting research..."):
            response = research_agent.run(latest_query)
            
            # Improved response extraction logic
            if isinstance(response, dict):
                agent_response = response.get("content", "No content available.")
            elif hasattr(response, "content"):
                agent_response = response.content
            else:
                agent_response = str(response)
            
            st.session_state.conversation.append({
                "role": "assistant",
                "message": agent_response
            })
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.conversation.append({
            "role": "assistant",
            "message": f"❌ Error processing request: {e}"
        })
    finally:
        st.session_state.loading = False
        st.session_state.processing = False
        st.rerun()
