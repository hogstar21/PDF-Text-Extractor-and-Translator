import argparse
import os
from pathlib import Path
import PyPDF2
import requests
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
    """Translate text using an offline approach with NLTK for basic Latin translation."""
    logger.info(f"Translating text to {target_language}")
    
    try:
        # Import offline translation tools
        import nltk
        from nltk.translate import Alignment
        
        # This is a very simplified approach for demo purposes
        # For actual Latin translation, you would need a proper Latin-English dictionary
        # or a specialized library
        
        # First, check if we have the necessary NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')
            
        # Basic Latin-English dictionary for common words
        latin_dict = {
            "et": "and", "in": "in", "est": "is", "ad": "to", "non": "not",
            "ex": "from", "cum": "with", "quod": "which/because", "sed": "but",
            "si": "if", "sunt": "are", "hoc": "this", "esse": "to be", "qui": "who",
            "ut": "as/that", "per": "through", "quam": "than/how", "aut": "or",
            "ego": "I", "tu": "you", "nos": "we", "vos": "you (plural)",
            "meus": "my", "tuus": "your", "noster": "our", "vester": "your (plural)",
            "homo": "man/human", "deus": "god", "vita": "life", "mors": "death",
            "terra": "earth/land", "aqua": "water", "ignis": "fire", "aer": "air",
            "rex": "king", "natura": "nature", "tempus": "time", "annus": "year",
            "dies": "day", "nox": "night", "sol": "sun", "luna": "moon",
            "caelum": "sky/heaven", "mare": "sea", "flumen": "river",
            "urbs": "city", "via": "way/road", "bellum": "war", "pax": "peace",
            "virtus": "virtue/courage", "amor": "love", "corpus": "body",
            "anima": "soul", "mens": "mind", "ratio": "reason", "scientia": "knowledge",
            "liber": "book/free", "verbum": "word", "nomen": "name",
            "magnus": "great/large", "parvus": "small", "bonus": "good", "malus": "bad",
            "verus": "true", "falsus": "false", "novus": "new", "vetus": "old",
            "facere": "to do/make", "videre": "to see", "dicere": "to say/speak",
            "audire": "to hear", "scire": "to know", "venire": "to come",
            "ire": "to go", "dare": "to give", "habere": "to have",
            "potestas": "power", "imperium": "command/empire"
        }
        
        # Tokenize the text
        sentences = nltk.sent_tokenize(text)
        translated_sentences = []
        
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            translated_words = []
            
            for word in words:
                # Remove punctuation for dictionary lookup
                clean_word = ''.join(c for c in word.lower() if c.isalnum())
                
                # Look up in dictionary, preserve case and punctuation
                if clean_word in latin_dict:
                    # Try to preserve the case
                    if word.isupper():
                        translated_words.append(latin_dict[clean_word].upper())
                    elif word[0].isupper():
                        translated_words.append(latin_dict[clean_word].capitalize())
                    else:
                        translated_words.append(latin_dict[clean_word])
                else:
                    # Keep original word if not in dictionary
                    translated_words.append(word)
            
            # Reconstruct the sentence - very simplistic approach
            translated_sentence = ' '.join(translated_words)
            
            # Some basic cleanup (this is overly simplistic for demo purposes)
            translated_sentence = translated_sentence.replace(' , ', ', ')
            translated_sentence = translated_sentence.replace(' . ', '. ')
            
            translated_sentences.append(translated_sentence)
        
        translated_text = ' '.join(translated_sentences)
        
        # Add translation note
        translation_note = (
            "\n\n[NOTE: This is a very basic translation attempt using a limited Latin-English dictionary. "
            "For accurate translations, please consult a professional translator or specialized Latin "
            "translation tools. Many words may remain untranslated or be translated incorrectly.]"
        )
        
        return translated_text + translation_note
        
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        # Instead of failing, return original with note about failure
        return text + "\n\n[ERROR: Translation failed. Please ensure you have an internet connection or try a different translation approach. Error details: " + str(e) + "]"

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
