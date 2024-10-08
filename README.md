# Chat_with_pdf
PDF Chat App is an interactive Streamlit web app for engaging with PDF content. Users upload PDFs, ask questions, and get answers using Hugging Face NLP models. The app features text extraction, dynamic querying, and chat history, all within a user-friendly interface. Ideal for exploring and querying document content.
# PDF Chat App

## Project Overview

The PDF Chat App is an interactive web application built with Streamlit, designed to enable users to interact with PDF documents in a conversational manner. Users can upload PDF files, ask questions about the content, and receive contextually relevant answers generated by state-of-the-art NLP models.

## Features

- **Upload PDF**: Seamlessly upload PDF files and extract their content.
- **Dynamic Querying**: Ask questions related to the uploaded PDF content.
- **NLP Integration**: Uses Hugging Face's transformers to provide accurate answers.
- **Chat History**: Keeps track of user queries and AI responses for reference.
- **Custom UI**: Provides an engaging and user-friendly interface with custom styling.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **PyPDF2**: For extracting text from PDF files.
- **Transformers**: For leveraging pre-trained NLP models to answer queries.
- **Langchain**: For text chunking and splitting.
- **Python**: Programming language used for development.

## Installation

To run this project, you need to install the required Python packages. You can do this using `pip`. First, ensure you have Python 3.7 or higher installed on your system. Then, install the necessary packages with the following command:

```bash
pip install streamlit pypdf2 transformers langchain
