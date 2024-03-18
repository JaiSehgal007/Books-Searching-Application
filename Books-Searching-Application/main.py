## Integrate our code OpenAI API
import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain import FewShotPromptTemplate

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key


## OPENAI LLMS
llm=OpenAI(temperature=0.4)

# streamlit framework
st.title('book Search Results')
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

    with st.expander('book Name'): 
        st.info(publishing_memory.buffer)

    with st.expander('Major Events'): 
        st.info(descr_memory.buffer)


# Streamlit Framework

st.title('Enter Genre for the book')
input_text1=st.text_input("It may be fantasy, fiction, horror...")

template1='''Find the book of '{genre}' genre and in {language} language'''
forth_input_prompt=PromptTemplate(
    input_variables=["genre","language"],
    template=template1,
)

print(forth_input_prompt.format(genre="fantasy",language="english"))

chain4=LLMChain(llm=llm,prompt=forth_input_prompt)

if input_text1:
    st.write(chain4({'genre':input_text1,'language':'english'}))


# Streamlit Framework
st.title('Search Genre for the book')
input_text2=st.text_input("Enter the book name...")

examples=[
    {"book":"The great gatsby","genre":"fiction"},
    {"book":"Treasure Islands","genre":"Adventure"},
]

example_formatter_template="""book: {book} genre: {genre}"""

fifth_input_prompt= PromptTemplate(
    input_variables=["book","genre"],
    template=example_formatter_template,
)

few_shot_prompt= FewShotPromptTemplate(
    examples=examples,
    example_prompt=fifth_input_prompt,
    prefix="Give the genre of each book\n",
    suffix="book: {input} genre: ",
    input_variables=["input"],
    example_separator="\n"
)

print(few_shot_prompt.format(input="Three Men in a boat"))

chain5=LLMChain(llm=llm,prompt=few_shot_prompt)

if input_text2:
    st.write(chain5({'input':input_text2}))
