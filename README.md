# LangChain-RAG for Financial and Business News Analysis
## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system for analyzing financial and business news over the past month. By leveraging LangChain, Named Entity Recognition (NER) and vector-based retrieval, our goal is to generate accurate, source-backed, and up-to-date information for users. This chatbos helps overcome key limitations of traditional large language models (LLMs), including lack of real-time knowledge, hallucination issues and lack of news source.

## 1. **Exploratory Data Analysis (EDA)**

- **Data Cleaning**:  
  - Preprocessed financial news data to remove unnecessary elements.  

- **Named Entity Recognition (NER) Analysis**:  
  - Generated a **word cloud** and **network graph** of extracted entities.  


## 2. **Sentiment Analysis with Fine-Tuned DistilBERT**:  
  - Fine-tuned **DistilBERT** on a financial [news sentiment dataset](https://github.com/Vindmn1234/Langchain-RAG-Project/blob/main/EDA%20%2B%20Sentiment%20fine-tuned%20model/all-data.csv). Improved accuracy from **0.31 to 0.85** ([details here](https://github.com/Vindmn1234/Langchain-RAG-Project/blob/main/EDA%20%2B%20Sentiment%20fine-tuned%20model/Fining-tuning.ipynb)).
  - Saved and Applied the model to classify sentiment in our dataset. ([shown in 3. Sentiment Analysis](https://github.com/Vindmn1234/Langchain-RAG-Project/blob/main/EDA%20%2B%20Sentiment%20fine-tuned%20model/Final%20Project-EDA.ipynb)).


## 3. Retrieval-Augmented Generation (RAG)
- **LangChain for RAG**
  - Utilizes **LangChain** to build a **Retrieval-Augmented Generation (RAG)** pipeline.
  - Handles **document processing, retrieval, and LLM integration**.

- **Named Entity Recognition (NER)**
  - Uses `Jean-Baptiste/roberta-large-ner-english` to extract key financial entities.

- **Embedding & FAISS Vector Storage**
  - Generates embeddings with `text-embedding-ada-002` via LangChain's `OpenAIEmbeddings`.
  - Stores embeddings in **FAISS** for fast retrieval.

- **Retrieval-Augmented Generation (RAG)**
  - Retrieves top **k** relevant news articles using **LangChain’s FAISS retriever**.
  - Uses **GPT-4 (via LangChain’s RetrievalQA)** to generate accurate, context-aware responses.

## 4. Evaluation Process

### **Evaluation Process**

- **Generate Q&A Pairs**  
  - GPT-4 generates **20 question-answer pairs** from retrieved financial news articles. These serve as the ground truth for evaluation. 

- **Test the RAG Model**  
  - Runs the RAG system on these queries and generated answers for the 20 questions. Compares responses to ground truth.  

- **Faithfulness Scoring**  
  - **GPT-4 acted as a judge**, scoring faithfulness (accuracy & correctness) on a **1-5 scale**. It also provided detailed feedback on mistakes and strengths.

### **Results**
- **Average Faithfulness Score:** **4.60/5**  
- **Most responses** rated **4-5**, indicating **high factual accuracy**.  
- **Minor inconsistencies** in some responses, but overall **strong performance**.  

✅ **Conclusion:**  
Our **RAG system significantly improves accuracy, transparency, and real-time relevance** compared to standard LLM outputs.


## 5. LangChain News Processing API ![](static/favicon.ico)

### Introduction

Welcome to the **LangChain News Processing API** – a cutting-edge, fast, and interactive API built using FastAPI and LangChain. This API leverages Retrieval-Augmented Generation (RAG) techniques to deliver real-time insights and summaries from a curated dataset of news articles.

### Key Features

- **Smart Retrieval:**  
  Perform an advanced similarity search among news articles. This feature can optionally filter results by date if a specific date is mentioned in the query, and uses a powerful language model to rank the most relevant news items.

- **Insight Retrieval:**  
  Obtain concise, analytical summaries of trends and key takeaways from the news articles without referencing individual sources. This provides a high-level overview of market sentiment and emerging trends.

### Data Source

The API processes data from CSV files that include essential fields such as the news publication date, the full text of the news article, and precomputed embeddings. Data is sourced from leading publications including:
- **The Wall Street Journal**
- **Bloomberg**
- **Financial Post**
- **The Verge**

*(Data used in this API covers the period from January to March in 2025.)* 
Example data can be found in [news_data](./service/news_data.csv/)

### How It Works

1. **Data Processing:**  
   The API reads a CSV file containing news articles, NER tags and GPT embeddings. It then creates LangChain Document objects and builds a FAISS vector store for efficient similarity search.

2. **Query Handling:**  
   Two types of queries are supported:
   - **Smart Retrieval:** Returns the most relevant news articles based on similarity search and LLM-based ranking.
   - **Insight Retrieval:** Summarizes the key trends and insights derived from the news articles.
   
3. **Interactive Interface:**  
   A simple web interface is provided, featuring a detailed welcome page and a query interface. This interface enables users to select the query type, enter their query, and view results—all styled for a user-friendly experience.

---

### Setting up the Virtual Environment
```bash
# Create the virtual environment:
python3.11 -m venv venv

# Activate the virtual environment:
source venv/bin/activate

# Install required packages:
python3 -m pip install -r requirements.txt
```
---

### Run the Application
```bash
# Run the application:
uvicorn app.main:app --reload
```
