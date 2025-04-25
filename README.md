# Text Summarizer

## Overview

The **Text Summarizer** is a Python-based tool designed to take a given text and generate a concise, summarized version of it. This project leverages **Natural Language Processing (NLP)** techniques, such as tokenization, lemmatization, and **TF-IDF** (Term Frequency-Inverse Document Frequency), to extract key points from longer text documents.

## Why This Project?

Text summarization has become increasingly important as the volume of text data grows in many domains, including news articles, scientific research, social media, and more. Summarizing long documents allows for faster understanding, easier information retrieval, and efficient decision-making. The primary goal of this project is to provide an automated solution for generating summaries from various text sources.

This tool can be useful for:
- Automatically summarizing research papers, articles, and reports.
- Assisting in text analysis tasks where quick understanding of content is needed.
- Reducing reading time for lengthy content while maintaining essential information.

## Setup and Usage Instructions

### 1. **Install dependencies**:
   To set up the environment, run:
   ```bash
   pip install -r requirements.txt
   ```

### 2. **Download NLTK datasets**
    In your code, run:
    ```bash
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    ```

### 3. **Run the script:**
   Go to text_summarizer_use.ipynb and try on your own inside the first cell with text you want to summary.

## Testing and Analyzation
To see results with the different number of returned summary and to see explanation, go to test_text_summarization.ipynb file.

## Contributing
- [Ostik24](https://github.com/Ostik24)
- [SofiiaPop](https://github.com/SofiiaPop)
- [Fenix125](https://github.com/Fenix125)
