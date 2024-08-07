import google.generativeai as genai
import pandas as pd
import os

proxies = {'http': 'http://172.24.25.11:8080', 'https': 'http://172.24.25.11:8080'}
os.environ["HTTP_PROXY"] = proxies['http']
os.environ["HTTPS_PROXY"] = proxies['https']

# Configure the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Define the model
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Define a combined prompt for sentiment analysis, categorization, and detailed analysis
def create_prompt(comment):
    prompt = f"""
    You are an expert in departmental categorizing, performing sentiment analysis, and providing in-depth analysis for comments received from customers.
    
    Additionally, the structure of Comment and Your response will be like below example, take comments from the input file.

    Comment1 : 
    Should add details about the transactions
    I have been using Alfa for several years now, and I am really happy with it, I have never run into any problems while using this and it always serves the purpose, my only concern is that it doesn’t provide any detail description about out transactions, it will just say member transfer for bank to bank transfer and POS merchant for every retail purchase with the card, however the other bank apps I use show details about the name and account number of the person you transferred the money to, and name and location of the shop you used your card at, and even the ATM location so you can clearly know what you used that money for, however with I actually have to jot everything down in my journal, the text messages we get does have details but we can’t really keep track of all the messages they are useful only at the moment, you get peace of mind that you payed exactly what you got, even if you don’t have the internet connection and as text messages from you guys have the details I don’t think it’s difficult for you to add in the app too, which is super useful and very essential for us. Hopefully you guys will look into it. Thanks

    Response : 
    Sentiment : Positive
    Department : Alfa App Department
    Analysis : The customer is generally happy with the bank's services but is dissatisfied with the lack of detailed descriptions for transactions in the app. They suggest including more specific transaction details like the name and account number of recipients, merchant names, and locations.
    
    Now analyze the following comment:
    {comment}
    
    Response:
    """
    return prompt

def get_gemini_response(prompt, comment):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content([prompt[0], comment])
    return response.text

# Generate responses for each comment
def analyze_comment(comment):
    prompt = create_prompt(comment)
    response_text = get_gemini_response(prompt, comment)
    
    sentiment, department, analysis = parse_response(response_text)
    
    return sentiment, department, analysis

# Parse the response to extract sentiment, category, and analysis
def parse_response(response_text):
    lines = response_text.strip().split('\n')
    sentiment = ""
    department = ""
    analysis = ""
    
    for line in lines:
        if line.startswith("Sentiment :"):
            sentiment = line.split(":")[1].strip()
        elif line.startswith("Department :"):
            department = line.split(":")[1].strip()
        elif line.startswith("Analysis :"):
            analysis = line.split(":")[1].strip()
    
    return sentiment, department, analysis

# Analyze all comments from a file
def analyze_comments_from_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("Comment,Sentiment,Department,Analysis\n")
        
        comment = ""
        for line in infile:
            line = line.strip()
            if line.startswith("Comment"):
                if comment:
                    # Process the previous comment block
                    sentiment, department, analysis = analyze_comment(comment)
                    outfile.write(f"\"{comment}\",\"{sentiment}\",\"{department}\",\"{analysis}\"\n")
                    comment = ""
            else:
                comment += " " + line
        
        if comment:
            # Process the last comment block
            sentiment, department, analysis = analyze_comment(comment)
            outfile.write(f"\"{comment}\",\"{sentiment}\",\"{department}\",\"{analysis}\"\n")

# Run the analysis
analyze_comments_from_file(r"D:\Genai\comments.txt", r"D:\Genai\analysis.csv")

# Display the results
results = pd.read_csv(r"D:\Genai\analysis.csv")
print(results)
