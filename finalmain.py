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
        doc = fitz.open(pdf_path)
        txt = ""
        for page in doc:
            txt += page.get_text()
        return txt
    except Exception as e:
        return f"error is {e} "

def summarize_long_text(text):
    """
    Improved summarization with better length control and faster processing
    """
    
    print("Loading summarization model...")
    # Use ONLY BART for both map and reduce - much faster and consistent
    pipeline_obj = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device_map="auto",  # Use "cpu" if no GPU
        max_length=1024,    # BART's max output length
        truncation=True
    )
    llm = HuggingFacePipeline(pipeline=pipeline_obj)

    # Calculate target length based on input
    word_count = len(text.split())
    print(f"Original text: {word_count} words")
    
    # Target 35-40% of original length
    target_words = int(word_count * 0.375)  # 37.5% of original
    min_target = max(100, int(target_words * 0.7))  # At least 70% of target
    max_target = min(800, int(target_words * 1.3))  # At most 130% of target
    
    print(f"Target summary length: {min_target}-{max_target} words")

    # Split text with larger chunks to preserve context
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        chunk_size=2000,  # Larger chunks
        chunk_overlap=200  # More overlap to preserve context
    )
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    print(f"Total chunks: {len(docs)}")

    # Improved prompts with explicit length requirements
    map_prompt_template = """
    Write a detailed summary of the following text. 
    Include all important points, key details, and main arguments.
    Make the summary substantial and informative - aim for about 40% of the original length.
    
    Text: "{text}"
    
    DETAILED SUMMARY:
    """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
    Below are detailed summaries from different sections of a document:
    "{text}"
    
    Combine these summaries into a comprehensive final summary that:
    1. Captures ALL major points and important details from each section
    2. Maintains logical flow and structure
    3. Preserves specific information, examples, and key insights
    4. Is substantial in length - aim for {target_length} words
    5. Does NOT oversimplify or lose important nuances
    
    Create a thorough, detailed final summary:
    """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, 
        input_variables=["text"],
        partial_variables={"target_length": f"{min_target}-{max_target}"}
    )

    # Load chain with same model for both steps (faster)
    print("Setting up summarization chain...")
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        verbose=True,
        # Remove token_max limitation - it was killing your output length!
        return_intermediate_steps=True  # This helps with debugging
    )
    
    # Run the chain
    print("Running summarization...")
    result = chain.invoke(docs)
    
    # Check if we need a second pass for length
    output_text = result['output_text']
    output_word_count = len(output_text.split())
    
    print(f"First pass output: {output_word_count} words")
    
    # If output is still too short, do a refinement pass
    if output_word_count < min_target * 0.8:  # If less than 80% of minimum target
        print("Output too short, doing refinement pass...")
        
        refinement_prompt = f"""
        The following summary is too brief and lacks detail. 
        Please expand it to be more comprehensive and detailed.
        Include more specific information, examples, and context.
        Target length: {min_target}-{max_target} words.
        
        Current summary: {output_text}
        
        EXPANDED DETAILED SUMMARY:
        """
        
        # Simple refinement using the same pipeline
        try:
            refined = pipeline_obj(
                refinement_prompt, 
                max_length=min(1024, max_target + 200),
                min_length=min_target,
                do_sample=False
            )[0]['summary_text']
            output_text = refined
        except:
            print("Refinement failed, using original output")
    
    final_word_count = len(output_text.split())
    print(f"Final summary: {final_word_count} words ({final_word_count/word_count*100:.1f}% of original)")
    
    return output_text

def simple_chunked_summarization(text):
    """
    Alternative approach: Progressive summarization with length preservation
    """
    print("Using alternative chunked approach...")
    
    # Initialize pipeline
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device_map="auto"
    )
    
    # Calculate target lengths
    word_count = len(text.split())
    target_ratio = 0.4  # 40% of original
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "? ", "! ", " "],
        chunk_size=1500,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(text)
    print(f"Processing {len(chunks)} chunks...")
    
    # Summarize each chunk with preserved length
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        chunk_words = len(chunk.split())
        target_length = max(50, int(chunk_words * target_ratio))
        
        try:
            summary = summarizer(
                chunk,
                max_length=min(target_length + 50, 400),
                min_length=max(30, target_length - 20),
                do_sample=False
            )[0]['summary_text']
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk {i+1}: {e}")
            # Fallback: take first part of chunk
            chunk_summaries.append(chunk[:500] + "...")
    
    # Combine all summaries
    combined_text = "\n\n".join(chunk_summaries)
    
    # Final summarization if combined text is too long
    combined_words = len(combined_text.split())
    if combined_words > word_count * 0.6:  # If still more than 60% of original
        print("Doing final consolidation...")
        final_target = max(200, int(word_count * target_ratio))
        
        try:
            final_summary = summarizer(
                combined_text,
                max_length=min(final_target + 100, 800),
                min_length=max(100, final_target - 50),
                do_sample=False
            )[0]['summary_text']
            return final_summary
        except:
            return combined_text
    
    return combined_text

if __name__ == "__main__":
    print("Opening file dialog to select PDFs...")
    root = TK()
    root.withdraw()
    pdf_files_to_summarize = askopenfilenames(
        title="Select PDF files to summarize",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
    )
    root.destroy()
    
    if not pdf_files_to_summarize:
        print("No files selected, exiting")
        exit()
    
    print(f"Found {len(pdf_files_to_summarize)} PDFs to summarize.")
    
    # Extract text from all PDFs
    all_text = ""
    for pdf_file in pdf_files_to_summarize:
        print(f"Extracting text from: {os.path.basename(pdf_file)}")
        text = extract_text_from_pdf(pdf_file)
        if text and not text.startswith("error"):
            all_text += text + "\n\n"
        else:
            print(f"Error extracting from {pdf_file}: {text}")
    
    if len(all_text.strip()) > 500:
        print("All text extracted. Starting summarization...")
        
        # Try the improved method first
        try:
            final_summary = summarize_long_text(all_text)
        except Exception as e:
            print(f"Main method failed: {e}")
            print("Trying alternative approach...")
            final_summary = simple_chunked_summarization(all_text)
        
        print("\n" + "="*80)
        print("FINAL SUMMARY:")
        print("="*80)
        print(final_summary)
        print("="*80)
        
        # Save to file
        with open("summary.txt", "w", encoding="utf-8") as f:
            f.write(final_summary)
        print("Summary saved to 'summary.txt'")
        
    else:
        print("Text too short for summarization. Please try larger documents.")