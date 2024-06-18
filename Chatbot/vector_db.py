from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel, ValidationError
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import json
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load documents
try:
    loader = PyPDFLoader('IITM BS Degree Programme - Student Handbook - Latest.pdf')
    documents = loader.load()
except Exception as e:
    logging.error(f"Error loading documents: {e}")
    documents = []

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Load embeddings
embeddings = HuggingFaceEmbeddings()
db = Chroma.from_documents(texts, embeddings)
print(db)

# Create retriever
retriever = db.as_retriever(search_kwargs={'k': 2})

# Initialize the LLM (Hugging Face Hub)
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceHub(
    huggingfacehub_api_token='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',
    repo_id=repo_id,
    model_kwargs={"temperature": 0.2, "max_new_tokens": 50}
)

# Attempt to load QA chain with error handling
try:
    qa_chain = load_qa_chain(llm, retriever)
except Exception as e:
    logging.error(f"Error initializing QA chain: {e}")
    qa_chain = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def get_answer():
    if qa_chain is None:
        return jsonify({'error': 'QA chain is not properly initialized'}), 500

    try:
        data = json.loads(request.data)
        query_request = QueryRequest(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        return jsonify({'error': 'Invalid input', 'details': str(e)}), 400

    query = query_request.query
    try:
        result = qa_chain({'question': query, 'chat_history': []})
        answer = result['answer']
        response = QueryResponse(answer=answer)
        return jsonify(response.dict())
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return jsonify({'error': 'An error occurred during question answering', 'details': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
