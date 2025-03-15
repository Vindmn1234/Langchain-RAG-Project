# app/main.py

from fastapi import FastAPI, HTTPException, Query, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from service import langchain_news

app = FastAPI(title="News Processing API")

# Mount the static directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to hold the vector store and QA chain once initialized
vector_db = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    """
    This event runs when the application starts. It processes the data 
    using the default CSV path and initializes the global variables.
    """
    global vector_db, qa_chain
    csv_path = "service/news_data.csv"  # default CSV path
    try:
        df = langchain_news.load_data(csv_path)
        # If embeddings exist in CSV, they should already be converted via load_data.
        # Otherwise, you can compute them here if needed.
        documents = langchain_news.create_documents(df)
        vector_db = langchain_news.create_vector_store(df, documents)
        qa_chain = langchain_news.setup_qa_chain(vector_db)
        print("Data processed and vector store created on startup.")
    except Exception as e:
        print(f"Error during startup data processing: {e}")

@app.get("/", response_class=HTMLResponse)
# Favicon image source: https://www.favicon.cc/?action=icon&file_id=819961
# Banner image source: https://www.shutterstock.com/search/latest-news-banner
def welcome():
    """
    A welcome page with a favicon, banner image, and detailed introduction.
    """
    html_content = """
    <html>
      <head>
        <title>Welcome to News Processing API</title>
        <!-- Favicon -->
        <link rel="icon" href="/static/favicon.ico" type="image/x-icon">
        <style>
          body {
            background-color: #f2f2f2;
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            margin: 0;
          }
          .container {
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            display: inline-block;
            max-width: 800px;
          }
          h1 {
            color: #333;
            margin-bottom: 20px;
          }
          p {
            color: #555;
            line-height: 1.6;
            margin-bottom: 20px;
          }
          .button {
            background-color: #007BFF;
            color: #fff;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-size: 16px;
            display: inline-block;
            margin-top: 20px;
          }
          .button:hover {
            background-color: #0056b3;
          }
          .banner {
            max-width: 100%;
            border-radius: 8px;
            margin-bottom: 20px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <!-- Banner Image -->
          <img src="/static/banner.jpg" alt="News Banner" class="banner">
          <h1>Welcome to News Processing API</h1>
          <p>Retrieve smart news insights and trends with ease.</p>
          <p>
            This is a LangChain-based RAG (Retrieval-Augmented Generation) news API designed to provide real-time analysis and summarization of financial and technology news.
            Our advanced system leverages similarity search and powerful language models to deliver relevant, insightful summaries and key takeaways.
          </p>
          <p>
            Data is sourced from leading publications such as The Wall Street Journal, Bloomberg, Financial Post, and The Verge (from January to March in 2025).
            Whether you're looking for smart retrieval of specific news events or a concise overview of current trends, our API is here to help.
          </p>
          <a class="button" href="/interface">Start Asking</a>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/status", response_class=HTMLResponse)
def status():
    """
    Check whether the global variables have been initialized.
    """
    if vector_db and qa_chain:
        return "<h2>Data is processed. The vector store and QA chain are initialized.</h2>"
    else:
        return "<h2>Data is not processed yet. Please process the data first.</h2>"

@app.get("/interface", response_class=HTMLResponse)
def query_interface():
    """
    Render an HTML interface that allows users to select the query style,
    enter their query, and submit it.
    """
    html_content = """
    <html>
      <head>
        <title>News Query Interface</title>
        <style>
          body {
            background-color: #f2f2f2;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
          }
          .container {
            max-width: 700px;
            margin: 50px auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
          }
          h1, h2 {
            text-align: center;
            color: #333;
          }
          p {
            font-size: 16px;
            line-height: 1.6;
            color: #555;
            margin-bottom: 20px;
          }
          form {
            display: flex;
            flex-direction: column;
          }
          label {
            margin-top: 15px;
            font-weight: bold;
          }
          select, textarea {
            padding: 10px;
            font-size: 16px;
            margin-top: 5px;
          }
          input[type="submit"] {
            margin-top: 20px;
            padding: 10px;
            font-size: 16px;
            background-color: #007BFF;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
          }
          input[type="submit"]:hover {
            background-color: #0056b3;
          }
          .back-link {
            display: inline-block;
            margin-top: 20px;
            text-decoration: none;
            color: #007BFF;
            font-weight: bold;
          }
          .back-link:hover {
            text-decoration: underline;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>News Query Interface</h1>
          <h2>Choose Your Query Type</h2>
          <p>
            <strong>Smart Retrieval:</strong> This option performs an advanced similarity search among the news articles.
            It optionally filters results by date if a specific date is mentioned in the query and uses a language model to rank the most relevant news items.
          </p>
          <p>
            <strong>Insight Retrieval:</strong> This option provides a concise summary of trends and key takeaways from the news articles.
            It does not reference specific sources but focuses on delivering an analytical overview of the current events and market sentiment.
          </p>
          <form action="/process-query" method="post">
            <label for="query_style">Select Query Type:</label>
            <select name="query_style" id="query_style">
              <option value="Smart Retrieval">Smart Retrieval</option>
              <option value="Insights Retrieval">Insight Retrieval</option>
            </select>
            <label for="query_text">Enter Your Query:</label>
            <textarea id="query_text" name="query_text" rows="4" cols="50"></textarea>
            <input type="submit" value="Submit Query">
          </form>
          <a class="back-link" href="/">&#8592; Back to Home</a>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/process-query", response_class=HTMLResponse)
def process_query(query_style: str = Form(...), query_text: str = Form(...)):
    """
    Process the query based on the selected query style (Smart Retrieval or Insight Retrieval)
    and return the result in an HTML page.
    
    The result is formatted with newline characters (\n) to improve readability.
    """
    global vector_db, qa_chain
    if not vector_db or not qa_chain:
        return HTMLResponse(
            content="<h2>Error: Data not processed yet. Please process data first.</h2>",
            status_code=400,
        )
    try:
        if query_style == "Smart Retrieval":
            result = langchain_news.smart_retrieval(query_text, vector_db)
        elif query_style == "Insights Retrieval":
            result = langchain_news.insight_retrieval(query_text, vector_db)
        else:
            return HTMLResponse(
                content="<h2>Error: Invalid query style selected.</h2>",
                status_code=400,
            )
        # Format the raw result with explicit newlines for better readability.
        formatted_result = result.replace("\n", "<br>")
        html_response = f"""
        <html>
          <head>
            <title>Query Result</title>
            <style>
              body {{
                background-color: #f2f2f2;
                font-family: Arial, sans-serif;
                padding: 20px;
              }}
              .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: #fff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
              }}
              h1 {{
                text-align: center;
                color: #333;
              }}
              p {{
                line-height: 1.6;
              }}
              a {{
                display: inline-block;
                margin-top: 20px;
                text-decoration: none;
                color: #007BFF;
              }}
              a:hover {{
                text-decoration: underline;
              }}
            </style>
          </head>
          <body>
            <div class="container">
              <h1>Query Result</h1>
              <p><strong>Query Style:</strong> {query_style}</p>
              <p><strong>Query:</strong> {query_text}</p>
              <p><strong>Response:</strong><br>{formatted_result}</p>
              <a href="/interface">Back to Query Interface</a>
            </div>
          </body>
        </html>
        """
        return HTMLResponse(content=html_response)
    except Exception as e:
        return HTMLResponse(
            content=f"<h2>Error processing query: {e}</h2>",
            status_code=500,
        )