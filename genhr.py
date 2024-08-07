# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:10:32 2024

@author: Admin-bafpk
"""
import streamlit as st
from PIL import Image  # Ensure Pillow library is installed
import os
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from autogen import AssistantAgent, UserProxyAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

# Proxy configuration
proxies = {'http': 'http://172.24.25.11:8080', 'https': 'http://172.24.25.11:8080'}
os.environ["HTTP_PROXY"] = proxies['http']
os.environ["HTTPS_PROXY"] = proxies['https']
# Configuration
FAISS_INDEX_PATH_1 = r'D:\Genai\faiss_index_hr'


FAISS_INDEX_PATH_2 = r'D:\Genai\faiss_index_hr_v2'
FAISS_INDEX_PATH_3 = r'D:\Genai\faiss_index_hr_v3'

ROBOT_IMAGE_PATH = r'D:\Genai\robo.png'  # Update this path to the location of your robot image
LOGO_PATH = r"D:\Genai\Bank_alfalah_logo.png"


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

def get_conversational_chain():
    prompt_template = """
       Answer the question as detailed as possible from the provided context or anything related from the sources. Answer the general information you have completely, do differentiate between the headers and its context like geographical context and if possible provide relevant information from which the question is asked for example sources of content, your task is to answer the question from sources of content wherever possible. If the complete answer is not in the provided context, say, "answer is not available in the context".
        
        

        The question may be in Roman Urdu. If the question is in Roman Urdu, translate the question to English and provide the answer in English. Additionally, if possible, also provide the answer in Roman Urdu based on the context or anything related.
        
        Also if there are greetings questions, kindly reply the answer in follow-up question of greetings. Do answer any question if related to general public questions from google search. Used google search engine to search the real time public data.

        If the question requires specific details not provided in the context (e.g., travel allowance,medical allowance based on grade), ask a follow-up question to gather the necessary information before providing the full answer.

        Context:
        {context}

        Question:
        {question}

        Answer in English:
        Answer in Roman Urdu (if possible):
        Follow-up Question (if needed):
        Document Name & Page No :
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
            logging.info("Loading FAISS index from paths: %s, %s, and %s",  FAISS_INDEX_PATH_1,FAISS_INDEX_PATH_2,FAISS_INDEX_PATH_3)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            vector_store_1 = FAISS.load_local(FAISS_INDEX_PATH_1, embeddings,allow_dangerous_deserialization=True)
            vector_store_2 = FAISS.load_local(FAISS_INDEX_PATH_2, embeddings,allow_dangerous_deserialization=True)
            vector_store_3 = FAISS.load_local(FAISS_INDEX_PATH_3, embeddings,allow_dangerous_deserialization=True)
            
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

    except Exception as e:
        logging.error(f"Error processing user input: {e}")
        st.error("An error occurred while processing your question. Please try again.")

def fallback_answer(question):
    try:
        prompt = f"""
 Answer the question as detailed as possible from the provided context or anything related from the sources. Answer the general information you have completely, do differentiate between the headers and its context like geographical context and if possible provide relevant information from which the question is asked for example sources of content, your task is to answer the question from sources of content wherever possible. If the complete answer is not in the provided context, say, "answer is not available in the context".
  
  

  The question may be in Roman Urdu. If the question is in Roman Urdu, translate the question to English and provide the answer in English. Additionally, if possible, also provide the answer in Roman Urdu based on the context or anything related.
  
  Also if there are greetings questions, kindly reply the answer in follow-up question of greetings. Do answer any question if related to general public questions from google search. Used google search engine to search the real time public data.

  If the question requires specific details not provided in the context (e.g., travel allowance,medical allowance based on grade), ask a follow-up question to gather the necessary information before providing the full answer.

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
                    <h1>BAFL GenHR in Roman Urdu</h1>
                    <p>Ask any HR Manual-related question in Roman Urdu or English</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Add robot image in the third column
        with cols[2]:
            my_robot = add_logo(ROBOT_IMAGE_PATH, width=150, height=100)
            st.image(my_robot, use_column_width=False)

    user_question = st.text_input("Your Question", placeholder="Type your question here...")

    if user_question:
        st.markdown(f"<div class='response-container'>Question: {user_question}</div>", unsafe_allow_html=True)
        user_input(user_question)

if __name__ == "__main__":
    main()
