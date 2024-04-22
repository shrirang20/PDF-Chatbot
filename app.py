import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import cassio
from PyPDF2 import PdfReader
import os
import time 

load_dotenv(Path(".env"))

st.set_page_config(page_title="Pdf Chatbot", layout="wide")
st.title("PDF Chatbot")

upload_file = st.file_uploader("Upload a PDF", type=['pdf'])

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "astra_vector_index" not in st.session_state:
    st.session_state.astra_vector_index = None

with st.spinner("Pdf is Processing..."):
    if upload_file:
        pdfreader = PdfReader(upload_file)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
        
        if not st.session_state.pdf_processed:
            cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"), database_id=os.getenv("ASTRA_DB_ID"))
            llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
            embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            astra_vector_store = Cassandra(
                    embedding=embedding,
                    table_name="pdf_chat",
                    session=None,  
                    keyspace=None,
                )

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
            )
            texts = text_splitter.split_text(raw_text)
            astra_vector_store.add_texts(texts[:50])

            st.session_state.astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
            st.session_state.pdf_processed = True

            st.success("PDF uploaded and processed successfully!")

st.sidebar.markdown("## Chatbot")
st.sidebar.info('**Step 1:** Upload any book, article, or any document in Pdf format')
st.sidebar.info('**Step 2:** Ask Question related to the uploaded pdf')
if upload_file:
    st.sidebar.success("PDF uploaded successfully!")
else:
    st.sidebar.info("Upload a PDF to start")

st.sidebar.markdown(' If you want to upload new pdf first refresh then upload')
st.sidebar.markdown(' If anything goes wrong do hard refresh by using **Shift** + **F5** key')
st.sidebar.markdown('##### contact if facing any problem: official.shriraang@gmail.com')


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input("What's up?")

if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    query_text = prompt.strip()

    if query_text.lower() == "quit":
        st.markdown("Thank you for joining in! Feel Free to ask anything realed to uploaded file!")
        st.stop()

    if st.session_state.astra_vector_index is not None:

        def typing_animation(text, speed):
            for char in text:
                yield char
                time.sleep(speed)
        
        answer = st.session_state.astra_vector_index.query(query_text, llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))).strip()
        typing_speed = 0.02

        with st.chat_message("assistant"):
            st.write_stream(typing_animation(answer,typing_speed))
        

        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Database not initialized. Kindly reload and upload the PDF first.")