import fitz
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    try:
        doc=fitz.open(pdf_path)
        txt=""
        for page in doc:
            txt+=page.get_text()
        return txt;
    except Exceptions as e:
        return f"ERror reading pdf: {e}"
    
def summarize_long_text(text):
    print("loading summarization model... thaam jao")
    summarizer=pipeline("summarization",model="facebook/bart-large-cnn")
    #splitting the text into smaller chunks
    desired_summary_ratio=0.1
    word_count=len(text.split())
    max_len=int(word_count*desired_summary_ratio)
    min_len=int(max_len*.5)
    max_len = min(max_len, 1024)
    max_len = max(max_len, 100) 
    min_len=max(min_len,50)
    sentence_separators = ["\n\n", "\n", ". ", "? ", "! ", " "]
    text_splitter=RecursiveCharacterTextSplitter(
        separators=sentence_separators,
        chunk_size=1000,
        chunk_overlap=50
    )
    chunks=text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks")
    print("now summarizing individual chunks ")
    chunk_summaries=[]
    
    for i , chunk in enumerate(chunks):
        print("summarizing chunk",i+1/len(chunks))
        print( "the chunk is ")
        print(chunk)
        #we set maxlength to be a fraction of the chunk size
        summary=summarizer(chunk,max_length=150,min_length=40,do_sample=False)
        print(" and the summary is ")
        print(summary[0]['summary_text'])
        chunk_summaries.append(summary[0]['summary_text'])
        
    print(" now combining all chunks")
    combined_summary=" ".join(chunk_summaries)
    final_summary=summarizer(combined_summary,max_length=max_len,min_length=min_len,do_sample=False)
    return final_summary[0]['summary_text']

if __name__ == "__main__":
    # 👇 Use a LONG document here (e.g., 10+ pages)
    pdf_file_path = r"C:\Users\Mayer\Downloads\Origins and Archaeological Evidence.pdf" 
    
    # Step 1: Extract text
    print("Step 1: Extracting text from PDF...")
    document_text = extract_text_from_pdf(pdf_file_path)
    
    if "Error" in document_text:
        print(document_text)
    elif len(document_text) < 1000:
        print("Document is too short for this script. Use the simple summarizer.")
    else:
        print("Text extraction successful.")
        
        # Step 2: Summarize the long text
        print("\nStep 2: Summarizing the long text...")
        final_summary = summarize_long_text(document_text)
        print("\n--- ✅ FINAL SUMMARY (from long document) ---")
        print(final_summary)