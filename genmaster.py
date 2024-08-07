import streamlit as st
from PIL import Image
import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from autogen import AssistantAgent, UserProxyAgent
from PyPDF2 import PdfReader
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)

# Proxy configuration
proxies = {'http': 'http://172.24.25.11:8080', 'https': 'http://172.24.25.11:8080'}
os.environ["HTTP_PROXY"] = proxies['http']
os.environ["HTTPS_PROXY"] = proxies['https']

# Configuration
FAISS_INDEX_PATH_1 = r'D:\Genai\faiss_index_ops_v1\faiss_index'
FAISS_INDEX_PATH_3 = r'D:\Genai\faiss_index_ops_v2\faiss_index'
FAISS_INDEX_PATH_2 = r'D:\Genai\faiss_index_ops'
ROBOT_IMAGE_PATH = r'D:\Genai\robo.png'  # Update this path to the location of your robot image
LOGO_PATH = r"D:\Genai\Bank_alfalah_logo.png"
UPLOAD_PATH = r'D:\Genai_ops\Retail Banking'  # Path to save the uploaded files

def apply_custom_css():
    st.markdown("""
        <style>
            .stApp {
                background-color: #7198ad;
            }
            .header-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                background-color: #7198ad;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .header-text {
                text-align: center;
                flex: 1;
            }
            .header-text h1 {
                font-size: 2.5em;
                color: #ffffff;
                margin: 0;
            }
            .header-text p {
                font-size: 1.2em;
                color: #dcdcdc;
                margin: 0;
            }
            .logo-container, .robot-container {
                flex-shrink: 0;
            }
            .stTextInput>div>div>input {
                color: #121111 !important;
            }
            .response-container {
                color: #dcdcdc !important;
                background-color: transparent;
                padding: 0;
                margin: 0;
                border: none;
                box-shadow: none;
            }
        </style>
    """, unsafe_allow_html=True)

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def get_pdf_text(file_path):
    all_texts = []
    try:
        pdf_reader = PdfReader(file_path)
        for page_number, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Check if the extracted text is not empty
                all_texts.append(page_text)
            else:
                logging.warning(f"No text found on page {page_number + 1} in {file_path}")
    except Exception as e:
        logging.error(f"Error reading PDF: {e}")
    return all_texts

def get_text_chunks(texts, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks

def append_to_vector_store(existing_vector_store_path, text_chunks, retry_attempts=3, delay_between_retries=5):
    for attempt in range(retry_attempts):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Load existing vector store
            existing_vector_store = FAISS.load_local(existing_vector_store_path, embeddings, allow_dangerous_deserialization=True)
            
            # Process text chunks in batches
            batch_size = 100
            for i in range(0, len(text_chunks), batch_size):
                batch_chunks = text_chunks[i:i+batch_size]
                
                # Create new vector store from batch chunks
                new_vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings)
                
                # Append new vectors to the existing vector store
                existing_vector_store.merge_from(new_vector_store)
            
            # Save the updated vector store
            existing_vector_store.save_local(existing_vector_store_path)
            
            logging.info(f"Vector store updated successfully at {existing_vector_store_path}")
            return
        except Exception as e:
            logging.error(f"Error updating vector store on attempt {attempt + 1}/{retry_attempts}: {e}")
            if attempt < retry_attempts - 1:
                logging.info(f"Retrying in {delay_between_retries} seconds...")
                time.sleep(delay_between_retries)
            else:
                logging.error("Failed to update vector store after multiple attempts.")
                raise

def get_conversational_chain():
    prompt_template = """
        Answer the question as detailed as possible from the provided context or anything related. Do check the relevant information from which the question is asked for example sources of content, your task is to answer the question from sources of content wherever possible. Also show us the relevant sources name with page number for all information which you know. If the complete answer is not in the provided context, say, "answer is not available in the context".
        
        

        The question may be in Roman Urdu. If the question is in Roman Urdu, translate the question to English and provide the answer in English. Additionally, if possible, also provide the answer in Roman Urdu based on the context or anything related.
        
        Also if there are greetings questions, kindly reply the answer in follow-up question of greetings. Do answer any question if related to general public questions from google search. Used google search engine to search the real time public data.
        
        Additionally, if i give you the website link your task is to scrape the information from that website.

        If the question requires specific details not provided in the context, ask a follow-up question and provide possible related options to gather the necessary information before providing the full answer.

        Context:
        {context}

        Question:
        {question}

        Answer in English:
        Answer in Roman Urdu (if possible):
        Follow-up Question (if needed):
        Document & Page No :
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logging.error(f"Error creating conversational chain: {e}")
        return None

def user_input(user_question):
    try:
        with st.spinner('Processing your question...'):
            logging.info("Loading FAISS index from paths: %s, %s, and %s", FAISS_INDEX_PATH_1, FAISS_INDEX_PATH_2, FAISS_INDEX_PATH_3)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            vector_store_1 = FAISS.load_local(FAISS_INDEX_PATH_1, embeddings, allow_dangerous_deserialization=True)
            vector_store_2 = FAISS.load_local(FAISS_INDEX_PATH_2, embeddings, allow_dangerous_deserialization=True)
            vector_store_3 = FAISS.load_local(FAISS_INDEX_PATH_3, embeddings, allow_dangerous_deserialization=True)
            
            logging.info("FAISS indices loaded successfully")

            docs_1 = vector_store_1.similarity_search(user_question)
            docs_2 = vector_store_2.similarity_search(user_question)
            docs_3 = vector_store_3.similarity_search(user_question)
            
            docs = docs_1 + docs_2 + docs_3  # Combine results from all indices
            logging.info("Similarity search completed")

            chain = get_conversational_chain()
            if chain:
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                if "answer is not available in the context" in response["output_text"]:
                    fallback_response = fallback_answer(user_question)
                    st.markdown(f"<div class='response-container'>Reply: {fallback_response}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='response-container'>Reply: {response['output_text']}</div>", unsafe_allow_html=True)
            else:
                st.error("Failed to create the conversational chain.")
    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        st.error("An error occurred while processing your question. Please try again.")

def fallback_answer(question):
    try:
        prompt = f"""
         Answer the question as detailed as possible from the provided context or anything related. Do check the relevant information from which the question is asked for example sources of content, your task is to answer the question from sources of content wherever possible. Also show us the relevant sources name with page number for all information which you know. The question may be in Roman Urdu. If the question is in Roman Urdu, translate the question to English and provide the answer in English. Additionally, if possible, also provide the answer in Roman Urdu based on the context or anything related.
        
        Also if there are greetings questions, kindly reply the answer in follow-up question of greetings. Do answer any question if related to general public questions from google search. Used google search engine to search the real time public data.
        
        Additionally, if i give you the website link your task is to scrape the information from that website.

        If the question requires specific details not provided in the context, ask a follow-up question and provide possible related options to gather the necessary information before providing the full answer.

        Question:
        {question}

        Answer in English:
        Answer in Roman Urdu (if possible):
        Follow-up Question (if needed):
        Document Name & Page No :
        """

        config_list = [{
            'api_type': 'google',
            'model': 'gemini-1.5-pro-latest',
            'api_key': os.getenv('GOOGLE_API_KEY'),
            'top_p': 0.5,
            'max_tokens': 2048,
            'temperature': 0.2,
            'top_k': 20,
            'prompt': prompt
        }]
        
        assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
        user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})
        user_proxy.initiate_chat(assistant, message=question)
        response = assistant.get_response()
        return response
    except Exception as e:
        logging.error(f"Error in fallback answer: {e}", exc_info=True)
        return "An error occurred while getting the fallback answer."

def process_uploaded_file(uploaded_file, index_path, chunk_size, chunk_overlap, index_name):
    file_path = os.path.join(UPLOAD_PATH, uploaded_file.name)
    
    # Save the uploaded file to the server
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Only process the file if it hasn't been processed yet
    processed_key = f"{uploaded_file.name}_{index_name}"
    if processed_key not in st.session_state.processed_files:
        raw_texts = get_pdf_text(file_path)
        if raw_texts:
            text_chunks = get_text_chunks(raw_texts, chunk_size, chunk_overlap)
            append_to_vector_store(index_path, text_chunks)
            st.session_state.processed_files[processed_key] = True
            st.success(f"Uploaded PDF processed and vector store for {index_name} updated successfully")
        else:
            st.error("No text extracted. Please ensure the PDF file contains text and try again.")
    else:
        if not st.session_state.processed_files[processed_key]:
            st.session_state.processed_files[processed_key] = True
            st.info("This file has already been processed.")

def main():
    st.set_page_config(page_title="BAFL GenOPS in Roman Urdu", layout="wide")
    
    # Apply custom CSS
    apply_custom_css()
    
    # Header with Logo and Robot Image
    with st.container():
        cols = st.columns([1, 4, 1])
        
        # Add logo image in the first column
        with cols[0]:
            my_logo = add_logo(LOGO_PATH, width=150, height=100)
            st.image(my_logo, use_column_width=False)
        
        # Add header text in the second column
        with cols[1]:
            st.markdown("""
                <div class="header-text">
                    <h1>BAFL GenOPS in Roman Urdu</h1>
                    <p>Ask any Retail Manual-related question in Roman Urdu or English</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Add robot image in the third column
        with cols[2]:
            my_robot = add_logo(ROBOT_IMAGE_PATH, width=150, height=100)
            st.image(my_robot, use_column_width=False)

    # File uploader in the sidebar
    st.sidebar.header("Upload a PDF file")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    # Check if the file has already been processed
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}

    if uploaded_file is not None:
        with st.spinner("Processing the uploaded PDF file..."):
            # Process for different indices based on their configurations
            process_uploaded_file(uploaded_file, FAISS_INDEX_PATH_1, chunk_size=500, chunk_overlap=50, index_name="Index 1")
            process_uploaded_file(uploaded_file, FAISS_INDEX_PATH_2, chunk_size=10000, chunk_overlap=1000, index_name="Index 2")
            process_uploaded_file(uploaded_file, FAISS_INDEX_PATH_3, chunk_size=1000, chunk_overlap=100, index_name="Index 3")

    user_question = st.text_input("Your Question", placeholder="Type your question here...")

    if user_question:
        st.markdown(f"<div class='response-container'>Question: {user_question}</div>", unsafe_allow_html=True)
        user_input(user_question)

if __name__ == "__main__":
    main()
