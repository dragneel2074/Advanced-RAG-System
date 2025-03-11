# Part 1: RAG Chatbot with Document Indexing, OCR, and Knowledge Graph

This application provides a chat interface with Retrieval-Augmented Generation (RAG) capabilities, document management, OCR support for images, and a Neo4j knowledge graph for message categorization.

## Features

- Upload and index PDF documents, text files, and images
- Advanced OCR support for extracting text from images using EasyOCR
- Document selection for targeted queries
- Chat history persistence with MongoDB
- Vector search powered by Chroma
- LLM integration with Ollama (llama3.2)
- Knowledge Graph for message categorization with Neo4j

## Prerequisites

### MongoDB

This application uses MongoDB to store chat history. Install MongoDB:

- **Windows**: Download from [MongoDB Website](https://www.mongodb.com/try/download/community)
- **Linux**: Follow [installation instructions](https://www.mongodb.com/docs/manual/administration/install-on-linux/)
- **MacOS**: `brew install mongodb-community`

### Neo4j

This application uses Neo4j for the knowledge graph categorization. To set up Neo4j:

1. **Install Neo4j Desktop**:
   - Download from [Neo4j Download Page](https://neo4j.com/download/)
   - Install and launch Neo4j Desktop

2. **Create a Database**:
   - Create a new project (or use an existing one)
   - Click "Add Database" → "Create a Local Database"
   - Set a name (e.g., "ChatbotKnowledgeGraph")
   - Set a password (remember this for your .env file)
   - Click "Create"

3. **Start the Database**:
   - Click "Start" on your newly created database
   - Wait for the database to start (status will change to "Started")

4. **Configure Connection**:
   - Note the connection URI (default: `neo4j://localhost:7687`)
   - Note the username (default: `neo4j`)
   - Set the password in your `.env` file

### EasyOCR Dependencies

The application uses EasyOCR for image text extraction. While the Python package will be installed via pip, you may need some additional dependencies:

- **All platforms**: The first run will automatically download the necessary OCR models
- **GPU acceleration** (optional): If you have a CUDA-compatible GPU:
  - Install appropriate CUDA toolkit and CuDNN for your system
  - Set `gpu=True` in the EasyOCR Reader initialization in app.py

### Ollama

The chatbot uses Ollama for local LLM capabilities. [Install Ollama](https://ollama.ai/download) and run:

```
ollama pull llama3.2
```

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your configuration:
   ```
   MONGODB_URI=mongodb://localhost:27017/
   NEO4J_URI=neo4j://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password_here
   ```

## Setup Neo4j Knowledge Graph

Before running the application, you should initialize the Neo4j knowledge graph schema:

1. Make sure Neo4j is running
2. Run the initialization script:
   ```
   python init_neo4j.py
   ```
3. To verify the Neo4j setup, run the test script:
   ```
   python test_neo4j.py
   ```

This creates the necessary node labels, relationships, and constraints for categorizing chat messages.

## Running the Application

1. Make sure MongoDB is running
2. Make sure Neo4j is running
3. Make sure Ollama is running with the llama3.2 and  all-minilm:latest model (Pull from ollama.com/models)
4. Start the application:
   ```
   streamlit run app.py
   ```
5. Access the application at http://localhost:8501

## Configuration

You can configure the connections using environment variables:
- Set `MONGODB_URI` to your MongoDB connection string
- Set `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` for Neo4j connection

## Troubleshooting Neo4j

If you encounter Neo4j relationship errors:

1. Check that Neo4j is running
2. Make sure you've run the `init_neo4j.py` script to create the schema
3. Verify your credentials in the `.env` file
4. If errors persist, try clearing your database and running `test_neo4j.py`

The knowledge graph requires specific relationship types (`BELONGS_TO` and `CATEGORIZED_UNDER`) to function correctly. The initialization script creates these relationships automatically.

## Usage

1. Upload documents through the sidebar
2. For images, EasyOCR will automatically extract text (first run may take longer as models download)
3. Ask questions in the chat interface
4. Select specific documents for targeted queries, or choose "None" for direct LLM queries
5. Create new chat sessions or browse previous conversations 

#Part 2: Ticket Classification with Langchain Tools

## Ticket Classification with Langchain Tools (main.py)

This module provides a web interface and backend for classifying support tickets using multiple approaches:

- **Fine-Tuned Classification**: Uses a pre-trained transformer model (via the `transformers` library) to classify support tickets. Selected and Finetuned "distilbert/distilbert-base-uncased" and saved the finetuned model in huggingface (Dragneel/ticket-classification-v1)
- **Rule-Based Classification**: Utilizes keyword matching to determine the ticket category.
- **Clustering-Based Classification**: Leverages a KMeans clustering model on sentence embeddings (extracted via `sentence-transformers` and normalized with `scikit-learn`) to predict ticket clusters.

### Features

- User-friendly UI built with Streamlit for inputting ticket text.
- Integration with a fine-tuned text classification pipeline.
- Rule-based and clustering-based alternatives for robust classification.
- Detailed logging and debugging outputs for model performance.

### Prerequisites

- **Python Packages**: Install required packages including `streamlit`, `transformers`, `scikit-learn`, `sentence-transformers`, and `numpy`.
- **Model Files**: Ensure that a KMeans model file (`kmeans_model.pkl`) is available for clustering-based classification.

### Running the Application

To start the ticket classification interface, run:

```
streamlit run main.py
```

The application will load the necessary models and display a UI for entering support ticket text and selecting a classification method.
