import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.newspaper4k import Newspaper4k
import os

# Set your API key
os.environ['GOOGLE_API_KEY'] = 'AIzaSyCoXQ7uq-mWXfwzXxhT5nV6cgZMTybVntw'

def initialize_agent():
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo(), Newspaper4k()],
        description="You are a senior NYT researcher writing an article on a topic.",
        instructions=[
            "For a given topic, search for the top 10 links.",
            "Then read each URL and extract the article text, if a URL isn't available, ignore it.",
            "Analyze and prepare an NYT-worthy article based on the information.",
        ],
        markdown=True,
        show_tool_calls=True,
        add_datetime_to_instructions=True,
    )

# Initialize the agent
agent = initialize_agent()

# Streamlit UI
st.title("NYT Research Assistant")
st.write("This app uses Phi's Gemini model to research topics and generate NYT-style articles.")

# User input
query = st.text_input("Enter a topic or query for research:")

if st.button("Generate Article"):
    if query:
        st.write("### Researching topic...")
        try:
            # Fetch and process the response without streaming
            response = agent.run(query)  # Use `run()` or `get_response()` instead of `print_response()`

            # Display the query and processed response
            st.markdown(f"**Search Query:** {query}")
            st.markdown(f"**Response:**")

            # Handle response content based on structure
            if isinstance(response, dict):
                st.markdown(response.get('content', "No content available."))
            elif hasattr(response, 'content'):
                st.markdown(response.content)
            else:
                st.markdown(str(response))  # Fallback to generic display
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query to proceed.")
