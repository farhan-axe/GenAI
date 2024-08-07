import streamlit as st
import pandas as pd
import pyodbc
import google.generativeai as genai
import os
from PIL import Image
import re

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

logo_path = r"D:\Genai\Bank_alfalah_logo.png"
my_logo = add_logo(logo_path=logo_path, width=270, height=150)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #Ffd9db ;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.image(my_logo, use_column_width=False)

# Setting up environment variables for proxies and Google credentials
proxies = {'http': 'http://172.24.25.11:8080', 'https': 'http://172.24.25.11:8080'}
os.environ["HTTP_PROXY"] = proxies['http']
os.environ["HTTPS_PROXY"] = proxies['https']

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

PROMPT_FILE = r'D:\Genai\prompt.txt'
SEPARATOR_LINE = "also the sql code should not have ``` in the beginning or end and sql word in output"

def get_next_example_number():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, 'r') as file:
            existing_content = file.read()

        # Find all example numbers in the file
        matches = re.findall(r'Example (\d+)', existing_content)
        if matches:
            last_number = max(int(num) for num in matches)
            return last_number + 1
    return 16

def save_prompt(question, response):
    example_number = get_next_example_number()
    formatted_prompt = (
        f"\nExample {example_number} - {question},\n"
        f"the SQL command will be something like this \"{response}\";"
    )

    # Read existing prompts and save before the separator line if it exists
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, 'r') as file:
            existing_content = file.read()

        if SEPARATOR_LINE in existing_content:
            content_before_separator = existing_content.split(SEPARATOR_LINE)[0]
            new_content = content_before_separator + formatted_prompt + '\n' + SEPARATOR_LINE
        else:
            new_content = existing_content + formatted_prompt + '\n'
    else:
        new_content = formatted_prompt + '\n' + SEPARATOR_LINE

    with open(PROMPT_FILE, 'w') as file:
        file.write(new_content)

# Function to get response from Gemini model
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(["\n".join(prompt), question])
    return response.text

def get_gemini_analyze(question1, prompt1):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content([prompt1, question1])
    return response.text

# Function to retrieve query from the database
def read_sql_query(sql, server, database, username, password):
    conn_str = (
        f'DRIVER={{SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password};'
    )
    connection = pyodbc.connect(conn_str)
    cursor = connection.cursor()
    cursor.execute(sql)
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    result = [dict(zip(columns, row)) for row in rows]
    df = pd.DataFrame(result)
    cursor.close()
    connection.close()
    return df

server_name = "pkpccs01orbit"
database = "GenAI_Test"
username = "amber"
password = "Digital12345!"

def load_prompts():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, 'r') as file:
            return [line.strip() for line in file.readlines()]
    return []

prompts = load_prompts()

sqlite_to_sqlserver_mapping = {
    "datetime('now')": "GETDATE()",
    "date(column)": "CAST(column AS DATE)",
    "time(column)": "CAST(column AS TIME)",
    "strftime('%Y', date)": "YEAR(date)",
    "strftime('%m', date)": "MONTH(date)",
    "strftime('%d', date)": "DAY(date)",
    "strftime('%H', date)": "DATEPART(hour, date)",
    "strftime('%M', date)": "DATEPART(minute, date)",
    "strftime('%S', date)": "DATEPART(second, date)",
    "julianday(date)": "DATEDIFF(day, '0001-01-01', date)",
    "||": "+ (for string concatenation)",
    "length(string)": "LEN(string)",
    "substr(string, start, length)": "SUBSTRING(string, start, length)",
    "lower(string)": "LOWER(string)",
    "upper(string)": "UPPER(string)",
    "trim(string)": "LTRIM(RTRIM(string))",
    "replace(string, from_str, to_str)": "REPLACE(string, from_str, to_str)",
    "sum(column)": "SUM(column)",
    "avg(column)": "AVG(column)",
    "count(column)": "COUNT(column)",
    "max(column)": "MAX(column)",
    "min(column)": "MIN(column)",
    "round(number, decimals)": "ROUND(number, decimals)",
    "abs(number)": "ABS(number)",
    "random()": "RAND() (Note: RAND() in SQL Server is not a row-wise function)",
    "CASE WHEN condition THEN result ELSE alternative END": "CASE WHEN condition THEN result ELSE alternative END",
    "sqlite_version()": "SELECT @@VERSION (Note: @@VERSION returns SQL Server version info)"
}

st.title("Let's Explore DBG through Gen-AI")

def add_to_chatbox(sender, message):
    if isinstance(message, pd.DataFrame):
        st.table(message.style.set_properties(**{'font-weight': 'bold', 'color': 'black'}))
    else:
        st.text(f"{sender}: {message}")

if 'result_rows' not in st.session_state:
    st.session_state['result_rows'] = pd.DataFrame()
if 'current_prompt' not in st.session_state:
    st.session_state['current_prompt'] = ""
if 'query_approved' not in st.session_state:
    st.session_state['query_approved'] = False

custom_css = """
<style>
    .question-label {
        font-weight: bold !important;
        margin: 0;
        padding: 0;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown('<p class="question-label">Enter your question:</p>', unsafe_allow_html=True)
question = st.text_input("")

if st.button("Submit"):
    add_to_chatbox("You", question)
    add_to_chatbox("Processing", "Query...")

    max_attempts = 20
    attempt = 0
    success = False

    while attempt < max_attempts and not success:
        try:
            response = get_gemini_response(question, prompts)
            response = response.replace("strftime('%Y-%m',", "MONTH(").replace("strftime('%Y',", "YEAR(").replace("strftime('%H',", "DATEPART(").replace("```", "").replace("```", "").replace('sql', "").replace("strftime('%m',", "MONTH(").replace("HOUR(", "DATEPART(").replace("DATE(", "CAST(").replace('LIMIT 1', '').replace("GETCAST(", "GETDATE(").replace("DATEPART(Transaction_Date", "DATEPART(HOUR,Transaction_Date")
            st.session_state['result_rows'] = read_sql_query(response, server_name, database, username, password)
            add_to_chatbox("Query Results", st.session_state['result_rows'])
            st.session_state['current_prompt'] = response  # Save only the SQL command
            add_to_chatbox("Query Generated", st.session_state['current_prompt'])
            success = True
        except Exception as e:
            error_message = str(e)
            attempt += 1
            for sqllite, sqlserver in sqlite_to_sqlserver_mapping.items():
                if sqllite in error_message:
                    st.error("Error encountered")
            if attempt >= max_attempts:
                st.error(f"Maximum attempts reached. Last error: {error_message}")

    if success:
        st.info("Results Fetched Successfully")
    else:
        st.info("Operation failed after retries.")

if st.session_state['current_prompt']:
    st.write("Was the query and response helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve"):
            save_prompt(question, st.session_state['current_prompt'])
            st.session_state['query_approved'] = True
            st.success("Prompt saved successfully!")
            st.session_state['current_prompt'] = ""
    with col2:
        if st.button("Disapprove"):
            st.warning("Prompt discarded.")
            st.session_state['current_prompt'] = ""

if st.session_state['query_approved']:
    prompt1 = """You are expert in data analysis, your task is to provide detailed insights from data"""

    if st.button("Analyze"):
        if not st.session_state['result_rows'].empty:
            string_representation = ", ".join([f"{column}: {', '.join(map(str, st.session_state['result_rows'][column]))}" for column in st.session_state['result_rows'].columns])
            result_rows_string = "Analyzing the data: " + string_representation
            analyzed_response = get_gemini_analyze(result_rows_string, prompt1)
            add_to_chatbox("Analysis", analyzed_response)
            add_to_chatbox("Query Results", st.session_state['result_rows'])
        else:
            st.error("No data available to analyze. Please submit a query first.")


