# IITM BS Degree Program Chatbot

This project involves creating a Retrieval-Augmented Generation (RAG) based chatbot designed to assist with queries related to the IITM BS Degree program in Data Science and Applications. The chatbot provides information for both prospective qualifier applicants and current students using data from the BS Programme website and the Student Handbook. This document outlines the steps to build and run the chatbot using specified models from Hugging Face, containerized with Docker.

## Project Structure

The project contains the following key components:

1. **Backend**: Handles the core logic, including embedding documents, querying ChromaDB, and interacting with the language model for responses.
2. **Frontend**: A simple conversational UI for users to interact with the chatbot.
3. **ChromaDB**: Stores the embedded documents for similarity matching.
4. **Docker**: Containerizes the application for easy deployment.

## Requirements

- Python 3.7+
- Docker
- Git

## Setup and Usage

### Step 1: Clone the Repository

Clone the GitHub repository using the provided URL:

```bash
git clone https://oauth2:github_pat_<your pat>@github.com/<username>/<repo>.git
cd <repo>
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the root directory and add your Hugging Face token:

```bash
echo "HF_TOKEN=<your huggingface token>" > .env
```

### Step 3: Prepare Data

Ensure you have the BS Programme website data and Student Handbook stored in the `./data` directory. The embeddings for these documents will be generated and stored in `./chromaDB`.

### Step 4: Generate Embeddings

Run the script to embed the documents and store them in ChromaDB:

```bash
python scripts/generate_embeddings.py
```

### Step 5: Build Docker Image

Build the Docker image using the provided `Dockerfile`:

```bash
docker build -t iitm-bs-chatbot .
```

### Step 6: Run Docker Container

Run the Docker container to start the server:

```bash
docker run -d -p 8080:8080 --env-file .env iitm-bs-chatbot
```

### Step 7: Access the Chatbot

Open your browser and navigate to `http://localhost:8080/` to access the chatbot UI.

## Directory Structure

```
.
├── chromaDB
│   ├── index
│   └── metadata
├── data
│   ├── handbook.pdf
│   ├── website.html
│   └── ...
├── scripts
│   └── generate_embeddings.py
├── src
│   ├── backend
│   │   ├── main.py
│   │   └── ...
│   ├── frontend
│   │   ├── index.html
│   │   └── ...
├── Dockerfile
├── .env
└── README.md
```

## Implementation Details

### Backend

The backend is implemented in `src/backend/main.py` and handles:

- Loading documents and generating embeddings using Hugging Face models.
- Storing embeddings in ChromaDB.
- Querying ChromaDB for similar documents based on user input.
- Generating responses using the specified language model.

### Frontend

The frontend is a simple HTML page located in `src/frontend/index.html` which allows users to interact with the chatbot.

### ChromaDB

ChromaDB is used to store and query document embeddings. The embeddings are generated and stored using the `generate_embeddings.py` script.

### Docker

The Dockerfile sets up the environment, installs dependencies, and starts the server to host the chatbot application.

## References

1. [Hugging Face Token](https://huggingface.co/docs/hub/security-tokens)
2. [Creating GitHub PAT](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token)
3. [Starter on ChromaDB](https://www.chromadb.com/docs/quickstart)
4. [Getting Started with Docker](https://docs.docker.com/get-started/)

## Conclusion

This project provides a foundational implementation of a RAG-based chatbot for the IITM BS Degree program. The setup ensures that the chatbot is able to handle queries effectively using the specified models and data sources. Future improvements can focus on enhancing the chatbot's response accuracy, robustness, and user interface.
