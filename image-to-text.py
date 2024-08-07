import os
from google.cloud import vision_v1 as vision
from PIL import Image
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import fitz

# Proxy configuration
proxies = {'http': 'http://172.24.25.11:8080', 'https': 'http://172.24.25.11:8080'}
os.environ["HTTP_PROXY"] = proxies['http']
os.environ["HTTPS_PROXY"] = proxies['https']

# Set the environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\Genai\credentials.json"

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()


def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    return images

def ocr_image(image_bytes):
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text found in the image.")
        return ""

    # Extract the first text which is the most comprehensive one
    extracted_text = texts[0].description
    return extracted_text

def create_text_pdf(output_path, texts):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    text_height = height - 40  # Starting height

    for text in texts:
        lines = text.split('\n')
        for line in lines:
            if text_height <= 40:  # Create a new page if text goes beyond the page height
                c.showPage()
                text_height = height - 40

            c.setFont("Helvetica", 12)
            c.drawString(40, text_height, line)
            text_height -= 15  # Move to the next line

    c.save()

def main():
    input_pdf_path = r'D:\Genai\PDF.pdf'
    output_pdf_path = r'D:\Genai\workPDF.pdf'

    # Extract images from the PDF
    images = extract_images_from_pdf(input_pdf_path)

    # Perform OCR on each image
    texts = [ocr_image(image_bytes) for image_bytes in images]

    # Create text-based PDF
    if any(texts):
        create_text_pdf(output_pdf_path, texts)
        print(f"Text-based PDF created at: {output_pdf_path}")
    else:
        print("No text extracted from the PDF.")

if __name__ == "__main__":
    main()
