import streamlit as st
import numpy as np
from PIL import Image
import io
import traceback

@st.cache_resource
def get_ocr_reader():
    """
    Initialize and cache the EasyOCR reader.
    This ensures it's only loaded once per session.
    """
    try:
        import easyocr
        print("Initializing EasyOCR reader...")
        reader = easyocr.Reader(['en'], gpu=False)  # Change to True for GPU acceleration
        print("EasyOCR reader initialized successfully")
        return reader
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        traceback.print_exc()
        return None

def extract_text_from_image(image_file):
    """
    Extract text from an image using EasyOCR
    
    Args:
        image_file: A file-like object containing image data
        
    Returns:
        str: Extracted text or error message
    """
    try:
        # Get OCR reader
        reader = get_ocr_reader()
        if reader is None:
            return "Error: OCR reader could not be initialized."
        
        # Convert to PIL Image
        image = Image.open(image_file)
        print(f"Image opened: {image.format}, size: {image.size}, mode: {image.mode}")
        
        # Convert to numpy array for OCR
        image_np = np.array(image)
        
        # Perform OCR
        print("Starting OCR processing...")
        results = reader.readtext(image_np)
        print(f"OCR processing complete. Found {len(results)} text regions.")
        
        # Extract text from results
        if not results:
            return "No text found in the image."
            
        extracted_text = ""
        for detection in results:
            text = detection[1]  # Format is: [bbox, text, confidence]
            extracted_text += text + " "
            
        # Preview the extracted text
        preview = extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
        print(f"Extracted text preview: {preview}")
        
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        traceback.print_exc()
        return f"Error: {str(e)}" 