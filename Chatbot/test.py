import os
import json
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
from functools import lru_cache
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load environment variables
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./test.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Rate Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Database models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    chats = db.relationship('ChatHistory', backref='user', lazy=True)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.String(200), nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Create the database and tables within the application context
with app.app_context():
    db.create_all()

# Attempt to load the PDF document
try:
    loader = PyPDFLoader("final.pdf")
    documents = loader.load()
    logger.info("PDF document loaded successfully")
except Exception as e:
    logger.error(f"Error loading documents: {e}")
    documents = []

# Split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Load embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db_chroma = Chroma.from_documents(texts, embeddings)
retriever = db_chroma.as_retriever(search_kwargs={"k": 5})

# Initialize the LLM (Hugging Face Hub)
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceHub(
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    repo_id=repo_id,
    model_kwargs={"temperature": 0.2, "max_new_tokens": 150},
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever, return_source_documents=True
)

@lru_cache(maxsize=1000)
def get_cached_answer(query):
    result = qa_chain({"question": query, "chat_history": []})
    answer = result.get("answer", "No answer found")
    return answer

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=150)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=8)])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already exists.')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            return 'Invalid credentials'
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/")
@login_required
def home():
    return render_template("index.html", username=current_user.username)

@app.route("/query", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def get_answer():
    logger.info("Received query request")
    try:
        data = json.loads(request.data.decode("utf-8"))
        logger.debug(f"Request data: {data}")

        query = data.get("query", "")
        if query:
            answer = get_cached_answer(query)
            marker = "Helpful Answer:"
            start_pos = answer.find(marker)
            if (start_pos != -1):
                answer = answer[start_pos + len(marker):].strip()
            else:
                answer = "Marker not found"
            
            # Store the query and response in the database
            chat_history = ChatHistory(user_id=current_user.id, query=query, response=answer)
            db.session.add(chat_history)
            db.session.commit()
            db.session.refresh(chat_history)
        else:
            answer = "No query provided"

        return jsonify({"answer": answer})

    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
        return jsonify({"error": "Invalid JSON"}), 400
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
