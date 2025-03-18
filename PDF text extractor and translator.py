import argparse
import os
import uuid
import json
from pathlib import Path
import PyPDF2
import requests
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path, page_range=None):
    """Extract text from a PDF file with optional page range."""
    logger.info(f"Extracting text from {pdf_path}")
    
    # Parse page range if provided
    pages_to_extract = []
    if page_range:
        logger.info(f"Using page range: {page_range}")
        ranges = page_range.split(',')
        for r in ranges:
            if '-' in r:
                start, end = map(int, r.split('-'))
                # Convert from 1-based to 0-based indexing
                pages_to_extract.extend(range(start-1, end))
            else:
                # Convert from 1-based to 0-based indexing
                pages_to_extract.append(int(r) - 1)
        logger.info(f"Will extract {len(pages_to_extract)} pages")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            # If no page range specified, extract all pages
            if not pages_to_extract:
                pages_to_extract = range(total_pages)
            
            # Filter out any page numbers that are out of range
            pages_to_extract = [p for p in pages_to_extract if 0 <= p < total_pages]
            num_pages = len(pages_to_extract)
            
            logger.info(f"PDF has {total_pages} total pages, extracting {num_pages} pages")
            print(f"\n===== PDF EXTRACTION PROGRESS =====")
            print(f"Starting extraction of {num_pages} pages from {pdf_path}")
            if page_range:
                print(f"Using page range: {page_range}")
            print(f"=====================================\n")
            
            # For time estimation
            start_time = time.time()
            
            for i, page_num in enumerate(pages_to_extract):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
                
                # Show progress every 10 pages or at beginning/end
                if (i + 1) % 10 == 0 or i == 0 or i == num_pages - 1:
                    # Calculate progress and estimated time
                    if i > 0:
                        elapsed_time = time.time() - start_time
                        pages_per_second = (i + 1) / elapsed_time
                        remaining_pages = num_pages - (i + 1)
                        estimated_seconds = remaining_pages / pages_per_second if pages_per_second > 0 else 0
                        
                        # Convert to a readable format
                        hours, remainder = divmod(estimated_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        time_format = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                        
                        progress_percent = ((i + 1) / num_pages) * 100
                        
                        print(f"Extracting PDF: {progress_percent:.1f}% complete")
                        print(f"Processed page {i + 1} of {num_pages} (actual page number: {page_num + 1})")
                        print(f"Estimated time remaining: {time_format}")
                        print(f"-----------------------------------")
                    
                logger.info(f"Processed page {page_num + 1}/{total_pages} (#{i + 1} in sequence)")
            
            total_time = time.time() - start_time
            print(f"\n===== PDF EXTRACTION COMPLETE =====")
            print(f"Extracted {num_pages} pages in {total_time:.1f} seconds")
            print(f"=====================================\n")
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        raise

def translate_text(text, target_language, source_language="auto"):
    """
    Translate text using a reliable local approach with optional online fallbacks.
    """
    logger.info(f"Translating from {source_language} to {target_language}")
    
    # Start with the most reliable method (offline translation)
    # then try online methods only if needed
    translation_methods = [
        translate_with_google_translate,
        translate_with_fallback_dictionary,
        translate_with_mymemory,
        translate_with_lingva,
        translate_with_deepl_free
    ]
    
    # Try each translation method until one succeeds
    for i, translation_method in enumerate(translation_methods):
        try:
            logger.info(f"Trying translation method {i+1}/{len(translation_methods)}")
            result = translation_method(text, source_language, target_language)
            if result:
                return result
        except Exception as e:
            logger.error(f"Translation method {i+1} failed: {str(e)}")
    
    # If all methods fail, return original text with an error note
    error_message = "\n\n[ERROR: All translation methods failed. Please check your internet connection or try again later.]"
    return text + error_message


def translate_with_google_translate(text, source_language, target_language):
    """
    Translate using Google Translate's unofficial API with parallel processing.
    """
    logger.info("Attempting translation with Google Translate")
    
    # Use reasonable chunk size
    CHUNK_SIZE = 1000
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    total_chunks = len(chunks)
    
    # Source language code mapping for Latin
    if source_language.lower() in ["latin", "la", "lat"]:
        source_language = "la"
    elif source_language == "auto":
        source_language = "auto"
    
    # Print initial information
    print(f"\n===== TRANSLATION PROGRESS =====")
    print(f"Starting translation with Google Translate")
    print(f"Total chunks to process: {total_chunks}")
    print(f"Using parallel processing for faster translation")
    print(f"=====================================\n")
    
    # For time estimation
    start_time = time.time()
    translated_chunks = [None] * total_chunks  # Pre-allocate list
    
    # Function to translate a single chunk
    def translate_chunk(args):
        i, chunk = args
        
        try:
            # Construct the Google Translate URL
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": source_language,
                "tl": target_language,
                "dt": "t",
                "q": chunk
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Google Translate returns a nested array
                result = response.json()
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                    # Extract all translated parts
                    translation = ""
                    for part in result[0]:
                        if part and len(part) > 0:
                            translation += part[0]
                    
                    return i, translation
                else:
                    logger.error(f"Unexpected Google Translate response structure for chunk {i+1}")
                    return i, None
            else:
                logger.error(f"Google Translate API error: {response.status_code} for chunk {i+1}")
                return i, None
        except Exception as e:
            logger.error(f"Google Translate error for chunk {i+1}: {str(e)}")
            return i, None
    
    # Process chunks in batches to avoid overwhelming the API
    from concurrent.futures import ThreadPoolExecutor
    
    # Use max 10 workers - adjust based on your connection and API limits
    max_workers = 10
    batch_size = 30  # Process 30 chunks at a time
    
    # Split into smaller batches to show progress
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch = [(i, chunks[i]) for i in range(batch_start, batch_end)]
        
        # Progress update for this batch
        progress_percent = (batch_start / total_chunks) * 100
        
        # Calculate estimated time based on progress so far
        if batch_start > 0:
            elapsed_time = time.time() - start_time
            chunks_per_second = batch_start / elapsed_time
            remaining_chunks = total_chunks - batch_start
            estimated_seconds = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0
            
            # Convert to a readable format
            hours, remainder = divmod(estimated_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_format = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            print(f"Progress: {progress_percent:.1f}% complete")
            print(f"Processing chunks {batch_start+1}-{batch_end} of {total_chunks}")
            print(f"Estimated time remaining: {time_format}")
            print(f"-----------------------------------")
        else:
            print(f"Starting with first batch of chunks (1-{batch_end})")
            print(f"-----------------------------------")
        
        # Process this batch in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(translate_chunk, batch))
            
            # Update the translated_chunks list
            for i, translation in results:
                if translation is not None:
                    translated_chunks[i] = translation
                else:
                    logger.error(f"Failed to translate chunk {i+1}")
                    # Use a placeholder for failed chunks
                    translated_chunks[i] = f"[ERROR: Translation failed for chunk {i+1}]"
        
        # Add a small delay between batches to avoid rate limiting
        time.sleep(2)
    
    # Check for any None values in the translated_chunks list
    if None in translated_chunks:
        logger.error("Some chunks failed to translate")
        # Replace any None values with error messages
        translated_chunks = [chunk if chunk is not None else "[TRANSLATION ERROR]" for chunk in translated_chunks]
    
    total_time = time.time() - start_time
    print(f"\n===== TRANSLATION COMPLETE =====")
    print(f"Processed {total_chunks} chunks in {total_time:.1f} seconds")
    print(f"Average speed: {total_chunks/total_time:.2f} chunks per second")
    print(f"===================================\n")
    
    return "".join(translated_chunks)

def translate_with_mymemory(text, source_language, target_language):
    """Translate using MyMemory API (free tier)."""
    logger.info("Attempting translation with MyMemory API")
    
    # Use larger chunks with POST method
    CHUNK_SIZE = 2000  # Increased since we're using POST now
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    translated_chunks = []
    
    # Source language code mapping for Latin
    if source_language.lower() in ["latin", "la", "lat"]:
        source_language = "la"
    
    # Use auto detection if source is auto
    if source_language == "auto":
        source_language = ""  # MyMemory uses empty string for auto detection
    
    # For time estimation
    start_time = time.time()
    total_chunks = len(chunks)
    
    # Print initial information
    print(f"\n===== TRANSLATION PROGRESS =====")
    print(f"Starting translation with MyMemory")
    print(f"Total chunks to process: {total_chunks}")
    print(f"Estimated time will appear after processing the first chunk...")
    print(f"=====================================\n")
    
    # Process chunks sequentially to avoid overwhelming the API
    for i, chunk in enumerate(chunks):
        # Progress and time estimation
        if i > 0:
            elapsed_time = time.time() - start_time
            chunks_per_second = i / elapsed_time
            remaining_chunks = total_chunks - i
            estimated_seconds = remaining_chunks / chunks_per_second if chunks_per_second > 0 else 0
            
            # Convert to a readable format
            hours, remainder = divmod(estimated_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_format = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            progress_percent = (i / total_chunks) * 100
            
            # Print progress information on new lines
            print(f"Progress: {progress_percent:.1f}% complete")
            print(f"Processing chunk {i+1} of {total_chunks}")
            print(f"Estimated time remaining: {time_format}")
            print(f"-----------------------------------")
            
            logger.info(f"Translating chunk {i+1}/{total_chunks} - " 
                        f"{progress_percent:.1f}% complete, estimated time remaining: {time_format}")
        else:
            logger.info(f"Translating chunk {i+1}/{total_chunks}")
        
        # Add small delays between chunks to respect rate limits
        if i > 0 and i % 5 == 0:  # Add delay every 5 chunks
            time.sleep(1)
        
        url = "https://api.mymemory.translated.net/get"
        params = {
            "q": chunk,
            "langpair": f"{source_language}|{target_language}",
            "de": "your.email@example.com"  # Change this to a real email for higher rate limits
        }
        
        try:
            # Use POST instead of GET for larger text chunks
            response = requests.post(url, data=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "responseData" in data and "translatedText" in data["responseData"]:
                    translated_chunks.append(data["responseData"]["translatedText"])
                else:
                    logger.error(f"Unexpected response structure: {data}")
                    return None
            else:
                logger.error(f"MyMemory API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"MyMemory chunk {i+1} error: {str(e)}")
            return None
    
    print("\n===== TRANSLATION COMPLETE =====")
    print(f"Processed {total_chunks} chunks")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print(f"===================================\n")
    
    return "".join(translated_chunks)

def translate_with_deepl_free(text, source_language, target_language):
    """Translate using DeepL free API simulator."""
    logger.info("Attempting translation with DeepL Free API simulator")
    
    # DeepL has a limit per request
    CHUNK_SIZE = 5000
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    translated_chunks = []
    
    # Map Latin language code
    if source_language.lower() in ["latin", "la", "lat"]:
        # DeepL doesn't support Latin, so we'll attempt with Italian as it's closest
        source_language = "it"
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        logger.info(f"Translating chunk {i+1}/{len(chunks)} with DeepL Free")
        
        # Add delays between chunks
        if i > 0:
            time.sleep(3)
        
        url = "https://api-free.deepl.com/v2/translate"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # For DeepL free API simulator
        url = "https://api.deepl.com/v2/translate"
        payload = {
            "text": [chunk],
            "target_lang": target_language.upper(),
        }
        
        if source_language and source_language != "auto":
            payload["source_lang"] = source_language.upper()
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if "translations" in data and len(data["translations"]) > 0:
                    translated_chunks.append(data["translations"][0]["text"])
                else:
                    logger.error(f"Unexpected DeepL response structure: {data}")
                    return None
            else:
                logger.error(f"DeepL API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"DeepL translation error: {str(e)}")
            return None
    
    return "".join(translated_chunks)

def translate_with_lingva(text, source_language, target_language):
    """Translate using Lingva Translate (free and open source)."""
    logger.info("Attempting translation with Lingva Translate")
    
    # Lingva can handle larger chunks
    CHUNK_SIZE = 10000
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    translated_chunks = []
    
    # Map Latin language code
    if source_language.lower() in ["latin", "la", "lat"]:
        source_language = "la"
    
    # Use auto if source is auto
    if source_language == "auto":
        source_language = "auto"
    
    # Process each chunk - Lingva is typically faster, so we use multiple instances
    # List of public Lingva instances - we'll try them in order
    lingva_instances = [
        "https://lingva.ml", 
        "https://lingva.pussthecat.org",
        "https://lingva.garudalinux.org"
    ]
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        logger.info(f"Translating chunk {i+1}/{len(chunks)} with Lingva")
        
        # Minimal delay between chunks for Lingva
        if i > 0:
            time.sleep(0.5)
        
        # Try different Lingva instances
        for instance in lingva_instances:
            try:
                url = f"{instance}/api/v1/{source_language}/{target_language}/{chunk}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                # Set a timeout to avoid waiting too long
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if "translation" in data:
                        translated_chunks.append(data["translation"])
                        # Successfully translated with this instance, break the instance loop
                        break
                    else:
                        logger.error(f"Unexpected Lingva response structure: {data}")
                        continue
                else:
                    logger.error(f"Lingva instance {instance} API error: {response.status_code}")
                    continue
            except Exception as e:
                logger.error(f"Lingva instance {instance} error: {str(e)}")
                continue
        else:
            # This executes if the for loop completes without a break (all instances failed)
            logger.error(f"All Lingva instances failed for chunk {i+1}")
            return None
    
    return "".join(translated_chunks)

def translate_with_fallback_dictionary(text, source_language, target_language):
    """Fallback method that uses a basic dictionary for common Latin words."""
    logger.info("Using fallback dictionary translation")
    
    # Only translate Latin to other languages with this fallback
    if source_language.lower() not in ["latin", "la", "lat", "auto", ""]:
        logger.error(f"Fallback dictionary only supports Latin source, not {source_language}")
        return None
    
    try:
        # Import offline translation tools
        import nltk
        
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
        
        # For non-English target languages, we'd need dictionaries for those languages
        # This is simplified to show the concept
        if target_language != "en" and target_language != "eng":
            translation_note = f"\n\n[NOTE: Fallback dictionary translation from Latin to {target_language} is not fully supported. Using Latin to English translation instead.]"
            logger.warning(f"Non-English target {target_language} in fallback dictionary, defaulting to English")
        else:
            translation_note = (
                "\n\n[NOTE: This is a basic translation using a limited Latin-English dictionary. "
                "Many words remain untranslated and grammar will be incorrect. "
                "For accurate translations, please use an online service when available.]"
            )
        
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
            
            # Reconstruct the sentence
            translated_sentence = ' '.join(translated_words)
            
            # Basic cleanup
            translated_sentence = translated_sentence.replace(' , ', ', ')
            translated_sentence = translated_sentence.replace(' . ', '. ')
            
            translated_sentences.append(translated_sentence)
        
        translated_text = ' '.join(translated_sentences)
        return translated_text + translation_note
        
    except Exception as e:
        logger.error(f"Error in fallback translation: {str(e)}")
        return None

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
    parser.add_argument("--source-language", "-sl", default="auto", help="Source language code (default: auto for automatic detection)")
    parser.add_argument("--target-language", "-t", default="en", help="Target language code (default: en for English)")
    parser.add_argument("--output-dir", "-o", default=script_dir, help="Output directory for text files (default: same directory as the script)")
    parser.add_argument("--save-original", "-s", action="store_true", help="Save the original text as well")
    parser.add_argument("--pages", "-p", help="Page range to process (e.g., '1-10' or '5,10,15-20')")
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file paths
        pdf_name = Path(args.pdf_path).stem
        # Add page range to filename if specified
        if args.pages:
            suffix = args.pages.replace(',', '_').replace('-', 'to')
            original_output_path = output_dir / f"{pdf_name}_pages{suffix}_original.txt"
            translated_output_path = output_dir / f"{pdf_name}_pages{suffix}_{args.target_language}.txt"
        else:
            original_output_path = output_dir / f"{pdf_name}_original.txt"
            translated_output_path = output_dir / f"{pdf_name}_{args.target_language}.txt"
        
        # Extract text from PDF with optional page range
        original_text = extract_text_from_pdf(args.pdf_path, args.pages)
        
        # Save original text if requested
        if args.save_original:
            save_to_file(original_text, original_output_path)
            print(f"Original text saved to: {original_output_path}")
        
        # Translate text
        translated_text = translate_text(original_text, args.target_language, args.source_language)
        
        # Save translated text
        save_to_file(translated_text, translated_output_path)
        
        logger.info("Process completed successfully")
        print(f"\nProcessing complete!")
        print(f"Translated file saved to: {translated_output_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
