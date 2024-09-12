# NOTE create virtual enviorment and install the required libraries
#langchain==0.0.154
#PyPDF2==3.0.1
#python-dotenv==1.0.0
#streamlit-extras
#torch
#transfromers

import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# CSS for interactive view
def add_custom_css():
    st.markdown("""
    <style>
        .header-title {
            color: #4CAF50;
            font-size: 2.5rem;
            text-align: center;
        }
        .sidebar-title {
            font-size: 1.5rem;
            color: #FF6347;
        }
        .file-uploader {
            background-color: #f0f2f6;
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .chunk-highlight {
            background-color: #f9f9f9;
            padding: 10px;
            border-left: 4px solid #FF6347;
            margin: 5px 0;
            border-radius: 5px;
        }
        .chat-history {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .question-text {
            color: #4CAF50;
            font-weight: bold;
        }
        .answer-text {
            color: #FF4500;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title('📄 PDF Chat App 💬', anchor=None)
    st.markdown('<p class="sidebar-title">Developed for Kodzera</p>', unsafe_allow_html=True)
    st.write("Upload a PDF and ask questions to interact with its content.")
    add_vertical_space(20)
    st.write('Crafted with ❤️ by Idris (Software Developer)')

def main():
    add_custom_css()

    st.markdown("<h1 class='header-title'>Chat With Your PDF</h1>", unsafe_allow_html=True)
    st.markdown("Upload a PDF file and interact with its content. The AI will provide answers to your questions based on the text from the PDF.")
    # Here we upload file
    pdf = st.file_uploader("📁 Upload your PDF file here", type='pdf', key='file_uploader')
# For keeping history of asked question 
    if 'history' not in st.session_state:
        st.session_state.history = []  # Stores query-answer pairs

    if pdf is not None:
        # File upload indicator
        with st.spinner("Extracting text from PDF..."):
            pdf_reader = PdfReader(pdf)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        #pipeline creation
        qa_pipeline = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad")

      
        query = st.text_input("💬 Ask a question about your PDF file")

        if query:
            st.write("Searching for answers...")
            # Search through chunks to find the most relevant answer
            answers = []
            for chunk in chunks:
                answer = qa_pipeline({
                    'question': query,
                    'context': chunk
                })
                answers.append(answer)
            
            # Get the best answer based on confidence scores
            if answers:
                best_answer = max(answers, key=lambda x: x['score'])
                st.write(f"### 🤖 Best Answer: {best_answer['answer']}")
                st.write(f"**Confidence Score:** {best_answer['score']:.2f}")

                # Update chat history in session state
                st.session_state.history.append({
                    'question': query,
                    'answer': best_answer['answer'],
                    'score': best_answer['score']
                })
            else:
                st.write("No answers found in the document.")

    # Display Chat History
    if st.session_state.history:
        st.write("### 📝 Chat History")
        for entry in st.session_state.history:
            st.markdown(f"<div class='chat-history'><span class='question-text'>You:</span> {entry['question']}<br><span class='answer-text'>AI:</span> {entry['answer']} (Score: {entry['score']:.2f})</div>", unsafe_allow_html=True)
            st.write("---")

if __name__ == '__main__':
    main()
# now for run the code 
# use virtual enviorment
#workon venv
# streamlit run app.py  Now run this command to sart the project
