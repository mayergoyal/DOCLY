import fitz  # This is the PyMuPDF library

def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts the text from all pages.

    Args:
        pdf_path (str): The full path to the PDF file.

    Returns:
        str: The concatenated text from all pages of the PDF.
              Returns an error message if the file cannot be opened.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text()
        return full_text
    except Exception as e:
        return f"Error opening or reading the PDF file: {e}"

# This is the main part of the script that runs when you execute it
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Change this to the actual path of a PDF file on your computer
    pdf_file_path = r"C:\Users\Mayer\Downloads\Coereport (1).pdf"
    
    print(f"Attempting to extract text from: {pdf_file_path}\n")
    
    extracted_text = extract_text_from_pdf(pdf_file_path)
    
    print("--- EXTRACTED TEXT ---")
    print(extracted_text)
    print("\n--- END OF TEXT ---")