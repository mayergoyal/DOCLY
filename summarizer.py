import fitz
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    try:
        doc=fitz.open(pdf_path)
        fulltext=""
        for page in doc:
            fulltext+=page.get_text()
        return fulltext
    except Exception as e:
        return f"ERror reading pdf: {e}"
    
def summarize_text(text):
    print("Loading summarization model ...thaam jao")
    summarizer=pipeline("summarization",model="facebook/bart-large-cnn")
    print("model loaded .. shuru kr rhe ab")
    #adding some parameters to control the output length
    #bart has max length of 1024 tokens 
    summary=summarizer(text[:1024]
                       , max_length=150,
                       min_length=40,
                       do_sample=False
                       )
    return summary[0]['summary_text']

if __name__=="__main__":
    pdf_path=r"C:\Users\Mayer\Downloads\Origins and Archaeological Evidence.pdf"
    print(f"Extracting text from {pdf_path}")
    text=extract_text_from_pdf(pdf_path)
    if text.startswith("ERror"):
        print(text)
    else:
        print("Text extracted. Now summarizing ...")
        summary=summarize_text(text)
        print("\n--- SUMMARY ---\n")
        print(summary)
        print("\n--- END OF SUMMARY ---\n")

