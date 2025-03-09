import traceback

def plain_text_output(file_obj):
    """
    Extract plain text from a PDF file.
    
    Args:
        file_obj: A file-like object containing PDF data
        
    Returns:
        str: Extracted text content
    """
    try:
        from pdftext import PDF
        
        # Parse the PDF content
        pdf = PDF(file_obj)
        
        # Extract text from all pages
        text_content = ""
        for page in pdf:
            text_content += page.text + "\n\n"
        
        # If no text was extracted, return an error message
        if not text_content.strip():
            print("No text content found in the PDF file.")
            return ""
        
        return text_content
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        traceback.print_exc()
        return "" 