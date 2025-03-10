import streamlit as st
import os
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k

# =============================================================================
# NYT Research Assistant Application
#
# This application is an AI-powered research assistant designed to simulate
# a senior NYT researcher. It accepts a research topic, searches for relevant
# articles using DuckDuckGo and Newspaper4k, and then uses Google Gemini to
# synthesize an NYT-style article.
#
# Below is the complete code with integrated documentation and a Customer 
# Help Guide available to users in the sidebar.
# =============================================================================

# -----------------------------
# Session State Initialization
# -----------------------------
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

# -----------------------------
# Sidebar Configuration
# -----------------------------
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

# Display the sidebar configuration
configure_sidebar()

# -----------------------------
# Customer Help Guide Section
# -----------------------------
with st.sidebar.expander("Customer Help Guide"):
    st.markdown("""
    # Customer Help Guide

    ## 1. Getting Started
    **API Key Requirement:**  
    Before you begin, you must provide a valid Google API key. You can obtain your key from [Google MakerSuite](https://makersuite.google.com/).

    **System Requirements:**  
    Ensure you have a stable internet connection and that your web browser supports Streamlit applications.

    ## 2. Navigating the Application
    **Sidebar Settings:**
    - **Google API Key Input:** Enter your API key in the sidebar. The input field is masked for security.
    - **Link Count Selector:** Adjust the number of articles (between 1 and 50) the application will analyze for each research query. The default value is set to 10.
    - **Clear Conversation:** Use the "🧹 Clear Conversation" button to reset the chat history if needed.

    **Main Interface:**
    - **Title & Caption:** The header clearly identifies the application as the "NYT Research Assistant" with a brief description of its investigative journalism focus.
    - **Conversation Display:** The chat history between you and the assistant is shown as a series of messages.
    - **Input Field:** At the bottom, you can enter your research topic or feedback.

    ## 3. Using the Application
    **Step-by-Step Workflow:**
    1. **Enter Your API Key:** Input your Google API key in the sidebar.
    2. **Set Research Parameters:** Adjust the number of links to analyze if desired.
    3. **Submit Your Query:** Type your research topic into the chat input field and press enter.
    4. **Processing Indicator:** A spinner and message ("🔍 Conducting research...") will display while your query is processed.
    5. **View the Response:** The assistant's response—an NYT-style article synthesized from the top search results—will appear in the conversation history.

    **Behind the Scenes:**  
    The app uses DuckDuckGo for web search and Newspaper4k for article extraction, combined with the Gemini language model to generate a research report.

    ## 4. Troubleshooting & Common Issues
    - **Missing API Key:**  
      If you do not provide an API key, the app will display an error ("🔑 API key required to continue") and halt further operations.
    - **Empty or Invalid Input:**  
      If you submit a blank or whitespace-only query, the application will warn you to "Please enter a valid message."
    - **Processing Errors:**  
      Any errors during the research process will be displayed in the conversation log. If errors persist, try clearing the conversation and re-entering your query.

    ## 5. Frequently Asked Questions (FAQ)
    - **Q: How do I obtain a Google API Key?**  
      A: Visit [Google MakerSuite](https://makersuite.google.com/) to create and retrieve your API key.
    - **Q: How many links does the application analyze?**  
      A: You can choose between 1 and 50 links. The default is set to 10.
    - **Q: What type of research output can I expect?**  
      A: The assistant generates an NYT-style article synthesized from the content of the top search results.
    - **Q: Is my conversation saved?**  
      A: Yes, the conversation history is maintained throughout your session. It will be cleared if you use the "Clear Conversation" button.
    - **Q: Who do I contact if I encounter issues?**  
      A: For technical support, refer to the support contact details provided in your deployment documentation or reach out to your system administrator.

    ## 6. Additional Tips
    - **Security Reminder:**  
      Keep your API key confidential. Do not share it or include it in public repositories.
    - **Experiment with Queries:**  
      Try different research topics to see how the assistant adapts its analysis.
    - **Performance Note:**  
      Processing times may vary based on the complexity of the query and the number of links analyzed.
    """)

# -----------------------------
# API Key Validation
# -----------------------------
if not st.session_state.apikey:
    st.error("🔑 API key required to continue")
    st.stop()

# -----------------------------
# Agent Initialization with Caching
# -----------------------------
@st.cache_resource(show_spinner=False)
def create_research_agent(api_key: str, link_count: int):
    os.environ["GOOGLE_API_KEY"] = api_key  # Consider using secrets in production
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo(), Newspaper4k()],
        description="You are a senior NYT researcher writing an article on a topic.",
        instructions=[
            f"first you will make a plan  of what to answer and generate a series of question for the given topic"
            "For a given topic, search for the top 20 links.",
            "Then read each URL and extract the article text. If a URL isn't available, ignore it.",
            "Analyze and prepare an NYT-worthy article based on the information.",
            "make sure everytime used ask question you have to do propersearch and don't give response by your own,"
        ],
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=3,
        duckduckgo_news=True
    )

research_agent = create_research_agent(st.session_state.apikey, st.session_state.link_count)

# -----------------------------
# Main Interface Setup
# -----------------------------
st.title("📰 NYT Research Assistant")
st.caption("Collaborative AI-powered research platform for investigative journalism")

# -----------------------------
# Display Conversation History
# -----------------------------
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# -----------------------------
# Handling User Input
# -----------------------------
if user_input := st.chat_input("Enter your research topic or feedback...", disabled=st.session_state.loading):
    if not user_input.strip():
        st.warning("Please enter a valid message")
        st.stop()
    
    st.session_state.conversation.append({"role": "user", "message": user_input})
    st.session_state.loading = True
    st.rerun()

# -----------------------------
# Processing the Agent's Response
# -----------------------------
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
