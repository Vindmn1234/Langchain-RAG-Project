# service/langchain_news.py

import pandas as pd
import ast
import re
import numpy as np
from collections import Counter
import openai
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# ----------------------------
# Model & API Key Initialization
# ----------------------------

# Set OpenAI API key
with open("service/api_key.txt", "r") as key_file:
    openai.api_key = key_file.read().strip()

# Initialize LLM and Embedding models for LangChain
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai.api_key)
embedding_model = OpenAIEmbeddings(openai_api_key=openai.api_key)

# ----------------------------
# Business Logic Functions
# ----------------------------
def convert_embedding(emb):
    # If the embedding is stored as a string, convert it using ast.literal_eval.
    if isinstance(emb, str):
        try:
            emb_list = ast.literal_eval(emb)
            return np.array(emb_list)
        except Exception as e:
            print(f"Error converting embedding: {e}")
            return np.array([])  # Return an empty array on failure
    # Otherwise, assume it is already in a suitable format.
    return emb

def load_data(csv_path):
    """
    Load data from a CSV file.
    """
    df = pd.read_csv(csv_path)
    if "embeddings" in df.columns:
        df["embeddings"] = df["embeddings"].apply(convert_embedding)
    return df

def create_documents(df):
    """
    Create LangChain Document objects for each row in the dataframe.
    """
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=f"Date: {row['date']}\nNews: {row['news']}\nNamed Entities: {row.get('NER_entities', 'None')}",
            metadata={"date": row["date"], "NER_entities": row.get("NER_entities", "None")}
        )
        documents.append(doc)
    return documents

def create_vector_store(df, documents):
    """
    Create a FAISS vector store using precomputed embeddings and texts.
    """
    # Prepare embedding vectors and corresponding texts
    embedding_vectors = [np.array(emb) for emb in df["embeddings"].values]
    texts = df["news"].tolist()
    text_embedding_pairs = list(zip(texts, embedding_vectors))
    
    # Create vector store with LangChain
    vector_db = FAISS.from_embeddings(text_embedding_pairs, embedding_model)
    # Optionally save the index locally for future use
    vector_db.save_local("faiss_index")
    return vector_db

def setup_qa_chain(vector_db):
    """
    Set up a RetrievalQA chain using the FAISS retriever.
    """
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def smart_retrieval(query, vector_db):
    """
    Perform smart retrieval:
      - Optionally filter retrieved documents by a date (if found in the query).
      - Use the LLM to rank the retrieved documents.
    """
    import re
    # Extract date (supports YYYY-MM-DD or YYYY-MM format)
    date_match = re.search(r"\b(202\d-\d{2}-\d{2})\b", query) or re.search(r"\b(202\d-\d{2})\b", query)
    search_date = date_match.group(0) if date_match else None

    retrieved_docs = vector_db.similarity_search(query, k=10)
    if search_date:
        retrieved_docs = [doc for doc in retrieved_docs if search_date in doc.metadata.get("date", "")]
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    ranked_response = llm.predict(f"""
    You are an AI news assistant. Analyze the retrieved news articles and return the most relevant ones.

    User Query: {query}

    Retrieved News Articles:
    {context}

    If the exact date '{search_date}' is not found, return the closest related news instead.
    """)
    return ranked_response

def insight_retrieval(query, vector_db):
    """
    Retrieve insights (key trends, takeaways) from news articles without referencing specific sources.
    """
    import re
    date_match = re.search(r"\b(202\d-\d{2}-\d{2})\b", query) or re.search(r"\b(202\d-\d{2})\b", query)
    search_date = date_match.group(0) if date_match else None

    retrieved_docs = vector_db.similarity_search(query, k=10)
    if search_date:
        retrieved_docs = [doc for doc in retrieved_docs if search_date in doc.metadata.get("date", "")]
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    insights = llm.predict(f"""
    You are an expert financial analyst. Based on the retrieved information, summarize the key insights concisely.

    User Query: {query}

    Retrieved News Articles:
    {context}

    **Instructions:**
    - Do NOT mention specific sources or articles.
    - Provide only insights, trends, and key takeaways.
    - Use an analytical and objective tone.

    Provide the response in a structured and concise format.
    """)
    return insights