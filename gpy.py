import fitz
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from tkinter import Tk as TK
from tkinter.filedialog import askopenfilenames

from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from PIL import Image

def extract_text_from_pdf(pdf_path):
    try:
        doc=fitz.open(pdf_path)
        txt=""
        for page in doc:
            txt+=page.get_text()
            
        return txt
    except Exception as e:
        return f"error is {e} "
'''  
def summarize_long_text(text):
    print("loading summarization model... thaam jao")
    summarizer=pipeline("summarization",model="facebook/bart-large-cnn")
    #splitting the text into smaller chunks
    desired_summary_ratio=0.45
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
        
        
        #we set maxlength to be a fraction of the chunk size
        summary=summarizer(chunk,max_length=150,min_length=40,do_sample=False)
        
        chunk_summaries.append(summary[0]['summary_text'])
        
    print(" now combining all chunks")
    combined_summary=" ".join(chunk_summaries)
    final_summary=summarizer(combined_summary,max_length=max_len,min_length=min_len,do_sample=False)
    return final_summary[0]['summary_text']
'''
def summarize_long_text(text):
    print("Loading Pegasus model...")
    summarizer = pipeline(
        "summarization",
        model="google/pegasus-xsum",
        device_map="auto"
    )

    # Split text into chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,  # Pegasus handles long chunks
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    print(f"Total chunks: {len(chunks)}")

    # Summarize each chunk
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        word_count = len(chunk.split())
        min_len = max(100, int(word_count * 0.3))   # ~30%
        max_len = max(200, int(word_count * 0.6))   # ~60%
        result = summarizer(chunk, min_length=min_len, max_length=max_len, do_sample=False)
        summaries.append(result[0]['summary_text'])

    # Join summaries
    final_summary = "\n\n".join(summaries)
    return final_summary
if __name__=="__main__":
    print("opening file dialog to slect pdfs ")
    root=TK()
    root.withdraw()  #hiding the main window ..just want the dialog
    pdf_files_to_summarize=askopenfilenames(
        title="select karo",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        
    )
    root.destroy()
    if not pdf_files_to_summarize:
        print("no file selected , exiting")
        exit()
    print(f"Found {len(pdf_files_to_summarize)} PDFs to summarize.")
    #sabse pehle toh lets consolidate whole text
    all_text=""
    for pdf_file in pdf_files_to_summarize:
        text=extract_text_from_pdf(pdf_file)
        if text:
            all_text+=text+"\n\n"
    if len(all_text.strip())>200:
        print("All text extracted. Starting final summarization...")
        final_summary=summarize_long_text(all_text)
        print("Final Summary:")
        print(final_summary)
    else:
        print("Text is too short for a long summary. Please try larger documents.")
    