## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key


## OPENAI LLMS
llm=OpenAI(temperature=0.4)

# streamlit framework
st.title('Book Search Results')
input_text=st.text_input("Search for a book...")

# Memory
book_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
publishing_memory = ConversationBufferMemory(input_key='book', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='publishing', memory_key='description_history')


# Prompt Template
first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="Tell me about the book {name}"
)

chain=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='book',memory=book_memory)

# Prompt Template
second_input_prompt=PromptTemplate(
    input_variables=['book'],
    template="when was {book} written"
)
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='publishing',memory=publishing_memory)


# Prompt Template
third_input_prompt=PromptTemplate(
    input_variables=['publishing'],
    template="Mention 5 major events happened around {publishing} in the world"
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)


parent_chain=SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['book','publishing','description'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Book Name'): 
        st.info(publishing_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)