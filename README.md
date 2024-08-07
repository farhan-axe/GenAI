**BAFL GenOPS in Roman Urdu**

BAFL GenOPS in Roman Urdu is a Streamlit application that allows users to ask questions related to the Retail Manual in Roman Urdu or English. The application utilizes the Google Generative AI for embedding and question answering, and FAISS for vector store management. It also supports document processing from uploaded PDF files and includes a custom CSS design for the user interface.
Table of Contents

    Features
    Installation
    Usage
    Configuration
    Contributing
    License

Features

    Custom CSS Design: Enhance the look and feel of the application with custom CSS.
    Question Answering: Ask any question related to the Retail Manual in Roman Urdu or English.
    PDF Processing: Upload and process PDF files to update the FAISS vector store.
    Proxy Configuration: Support for HTTP/HTTPS proxies.
    Fallback Answering: Fallback mechanism for providing answers using Google search.

Installation

    Clone the repository:

    sh

git clone https://github.com/your-username/bafl-genops-roman-urdu.git
cd bafl-genops-roman-urdu

Install the required packages:

sh

    pip install -r requirements.txt

    Set up environment variables:
        GOOGLE_API_KEY: Your Google API key for accessing Google Generative AI services.

    Create necessary directories:
    Ensure the paths specified for the FAISS indices and image files exist or update them in the script.

Usage

    Run the Streamlit application:

    sh

    streamlit run app.py

    Access the application:
    Open your web browser and go to http://localhost:8501.

    Ask questions:
        Enter your question in the input box and press Enter.
        View the response in the application interface.

    Upload PDF files:
        Use the file uploader in the application to upload PDF files.
        The text from the PDF will be extracted, chunked, and appended to the FAISS vector store.

Configuration
Paths

    FAISS Index Paths:
        FAISS_INDEX_PATH_1
        FAISS_INDEX_PATH_2
        FAISS_INDEX_PATH_3

    Image Paths:
        ROBOT_IMAGE_PATH
        LOGO_PATH

Proxy Configuration

If you are behind a proxy, update the proxy settings in the script:

python

proxies = {'http': 'http://your-proxy-address:port', 'https': 'http://your-proxy-address:port'}
os.environ["HTTP_PROXY"] = proxies['http']
os.environ["HTTPS_PROXY"] = proxies['https']

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
License

This project is licensed under the MIT License.
