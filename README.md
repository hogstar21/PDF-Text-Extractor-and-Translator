# PDF Translator Tool

This Python tool extracts text from PDF files and translates it to your desired language.

## Features

- Extract text from PDF files
- Translate the extracted text to any language supported by Google Translate
- Save both the original text and the translated text as separate files
- Simple command-line interface

## Installation

1. Make sure you have Python 3.6+ installed
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python pdf_translator.py my_document.pdf
```

This will extract text from `my_document.pdf` and translate it to English (default).

### Options

- `-t, --target-language`: Target language code (default: "en" for English)
- `-o, --output-dir`: Output directory for text files (default: same directory as the script)
- `-s, --save-original`: Save the original text as well

### Examples

Translate a PDF to Spanish and save the original text:

```bash
python pdf_translator.py my_document.pdf -t es -s
```

Translate a PDF to French and save the files to a specific directory:

```bash
python pdf_translator.py my_document.pdf -t fr -o ./translations -s
```

## Supported Languages

This version of the tool includes a basic Latin-to-English dictionary for simple translations. It works entirely offline and doesn't require an internet connection.

**Important note about the translation quality**:
- The translation is very basic and uses a limited dictionary of common Latin words
- Many words may remain untranslated, especially specialized or uncommon terms
- The grammar and word order will often be incorrect
- For accurate translations, consider using a professional translator or specialized Latin translation tools

**Target Language**:
Currently, only translation to English (`en`) is supported in this offline version. Other target language codes will be ignored.

## Troubleshooting

If you encounter issues with the translation API, try:

1. Checking your internet connection
2. Breaking very large PDFs into smaller files
3. Waiting a bit and trying again (API rate limits)

## License

This tool is provided as-is under the MIT License.
