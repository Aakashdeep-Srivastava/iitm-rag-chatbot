from flask import Flask, request, jsonify, render_template
from pydantic import BaseModel
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

# Initialize Flask app
app = Flask(__name__)

# Load documents
loader = PyPDFLoader(r'C:\Users\Harsh Singh\Downloads\Chatbot\data\IITM BS Degree Programme - Student Handbook - Latest.pdf')
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Load embeddings
embeddings = HuggingFaceEmbeddings()
db = Chroma.from_documents(texts, embeddings)

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
except TypeError as e:
    print(f"Error: {e}")
    # Handle error gracefully or provide additional debugging information

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def get_answer():
    data = request.json
    query_request = QueryRequest(**data)
    query = query_request.query

    result = qa_chain({'question': query, 'chat_history': []})
    answer = result['answer']

    response = QueryResponse(answer=answer)
    return jsonify(response.dict())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
