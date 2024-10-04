import os

import streamlit as st
from dotenv import load_dotenv
from swarmauri.standard.agents.concrete.SimpleConversationAgent import \
    SimpleConversationAgent
from swarmauri.standard.conversations.concrete.MaxSystemContextConversation import \
    MaxSystemContextConversation
from swarmauri.standard.llms.concrete.GroqModel import GroqModel
from swarmauri.standard.messages.concrete.SystemMessage import SystemMessage

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
print(f"Loaded API Key: {API_KEY}")

# Initialize the Groq model
llm = GroqModel(api_key=API_KEY)
allowed_models = llm.allowed_models

# Create a conversation object
conversation = MaxSystemContextConversation()

def load_model(selected_model):
    """Load the model based on the selected model."""
    return GroqModel(api_key=API_KEY, name=selected_model)

def converse(input_text, system_context, model_name):
    """Process the user input and system context."""
    st.write(f"System context: {system_context}")
    st.write(f"Selected model: {model_name}")
    
    # Load the model
    llm = load_model(model_name)
    
    # Create the conversation agent
    agent = SimpleConversationAgent(llm=llm, conversation=conversation)
    agent.conversation.system_context = SystemMessage(content=system_context)
    
    # Process the user input
    input_text = str(input_text)
    st.write("Conversation history:", conversation.history)
    
    # Generate the result from the conversation agent
    result = agent.exec(input_text)
    st.write("Result:", result)
    
    return str(result)

# Streamlit interface for the chatbot
st.title("Chatbot with Swarmauri's Groq Model")
st.write("Interact with the agent using a system context and selected model")

# Input fields for system context and model selection
system_context = st.text_input("System Context", "Provide the system context here...")
model_name = st.selectbox("Model Name", allowed_models)

# Textbox for user input
input_text = st.text_area("Your Input", "Enter your conversation here...")

if st.button("Submit"):
    result = converse(input_text, system_context, model_name)
    st.write("Chatbot Response:", result)
