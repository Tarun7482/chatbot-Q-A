import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import faiss
from openai import OpenAI
from InstructorEmbedding import INSTRUCTOR




def get_pdf_data(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        number_of_pages = len(reader.pages)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
     text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
     )
     chunks=text_splitter.split_text(text)
     return chunks

def get_vector_store(chunk_text):
     #client=OpenAI()
     model= INSTRUCTOR('hkunlp/instructor-xl')
     embeddings=HuggingFaceInstructEmbeddings()
     vector_store= faiss.from_texts(texts=chunk_text, embedding=embeddings)
     return vector_store



def main():
    load_dotenv()
    st.set_page_config(page_title="chat with Pdf",page_icon="books")
    st.header("chat with PDFs")
    pdf_docs = st.file_uploader("upload your document", accept_multiple_files=True)
    if st.button("process"):
            with st.spinner("processing your document"):
                #get pdf raw data in text
                raw_text=get_pdf_data(pdf_docs)
                #display the text
                #st.write( raw_text)
                
                chunk_text=get_text_chunks(raw_text)
                st.text("chunks is going to executed")
                #st.write(chunk_text)
                vector_store=get_vector_store(chunk_text)

                #sst.write(vector_store)
    #st.text_input("Ask you question from your document")
    st.chat_input("Enter you query")

    add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
    )

   # Using "with" notation
    with st.sidebar:
     st.chat_input(f"enter your {add_selectbox}")

     st.link_button("HOME","http://localhost:8502/") 
     st.link_button("About us","https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/") 
     st.link_button("Documentaion","https://docs.streamlit.io/") 
     st.link_button("contact us","http://127.0.0.1:5500/aboutus.html") 

                


if __name__=='__main__':
    main()
