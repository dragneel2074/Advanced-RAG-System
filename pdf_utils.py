import traceback
from pdftext.extraction import plain_text_output

def pdf_output(file_obj):
    """
    Extract plain text from a PDF file.
    
    Args:
        file_obj: A file-like object containing PDF data
        
    Returns:
        str: Extracted text content
    """
    try:
        
        # Parse the PDF content
        pdf = plain_text_output(file_obj)
        print("PDF:")
        print(pdf)
        # # Extract text from all pages
        # text_content = ""
        # for page in pdf:
        #     text_content += page.text + "\n\n"
        
        # If no text was extracted, return an error message
        if not pdf.strip():
            print("No text content found in the PDF file.")
            return ""
        
        return pdf
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        traceback.print_exc()
        return "" 