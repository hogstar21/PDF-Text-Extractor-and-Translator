import argparse
import os
from pathlib import Path
import PyPDF2
from googletrans import Translator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logger.info(f"Extracting text from {pdf_path}")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            logger.info(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                logger.info(f"Processed page {page_num + 1}/{num_pages}")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise

def translate_text(text, target_language):
    """Translate text to the target language."""
    logger.info(f"Translating text to {target_language}")
    
    try:
        translator = Translator()
        
        # Break the text into chunks to avoid hitting API limits
        # Google Translate API typically has a limit around 5000 characters
        chunk_size = 4000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        translated_chunks = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Translating chunk {i+1}/{len(chunks)}")
            translation = translator.translate(chunk, dest=target_language)
            translated_chunks.append(translation.text)
        
        return "".join(translated_chunks)
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        raise

def save_to_file(text, output_path):
    """Save text to a file."""
    logger.info(f"Saving text to {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        logger.info(f"Text saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving text: {str(e)}")
        raise

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="Extract and translate text from PDF files")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--target-language", "-t", default="en", help="Target language code (default: en for English)")
    parser.add_argument("--output-dir", "-o", default=script_dir, help="Output directory for text files (default: same directory as the script)")
    parser.add_argument("--save-original", "-s", action="store_true", help="Save the original text as well")
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file paths
        pdf_name = Path(args.pdf_path).stem
        original_output_path = output_dir / f"{pdf_name}_original.txt"
        translated_output_path = output_dir / f"{pdf_name}_{args.target_language}.txt"
        
        # Extract text from PDF
        original_text = extract_text_from_pdf(args.pdf_path)
        
        # Save original text if requested
        if args.save_original:
            save_to_file(original_text, original_output_path)
        
        # Translate text
        translated_text = translate_text(original_text, args.target_language)
        
        # Save translated text
        save_to_file(translated_text, translated_output_path)
        
        logger.info("Process completed successfully")
        print(f"\nProcessing complete!")
        print(f"Translated file saved to: {translated_output_path}")
        if args.save_original:
            print(f"Original text saved to: {original_output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())