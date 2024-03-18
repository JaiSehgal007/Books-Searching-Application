# Conversational chatbot
import os
import streamlit as st
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI

# Streamlit UI
st.set_page_config(page_title="Appa Bot")
st.header("Hey, Let's Chat")

# Loading env
from dotenv import load_dotenv
load_dotenv()

# Initializing the LLM
os.environ["OPEN_API_KEY"]=os.getenv("OPENAI_KEY")
chat=ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"],temperature=0.7)

# Initialize the session state
if 'message_flow' not in st.session_state:
    st.session_state['message_flow']=[
        SystemMessage(content="Act as a AI senior software engineer")
    ]

# Function top load OpenAI messages
def get_openai_response(question):
    st.session_state['message_flow'].append(HumanMessage(content=question))
    answer=chat(st.session_state['message_flow'])
    st.session_state['message_flow'].append(AIMessage(content=answer.content))

    return answer.content


input=st.text_input("Input: ",key="input")
response=get_openai_response(input)

submit=st.button("Ask any question")

# If button is clicked
if submit:
    st.subheader("The response is")
    st.write(response)