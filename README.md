# RAG Chatbot with Document Indexing and OCR

This application provides a chat interface with Retrieval-Augmented Generation (RAG) capabilities, document management, and OCR support for images.

## Features

- Upload and index PDF documents, text files, and images
- Advanced OCR support for extracting text from images using EasyOCR
- Document selection for targeted queries
- Chat history persistence with MongoDB
- Vector search powered by Chroma
- LLM integration with Ollama (llama3.2)

## Prerequisites

### MongoDB

This application uses MongoDB to store chat history. Install MongoDB:

- **Windows**: Download from [MongoDB Website](https://www.mongodb.com/try/download/community)
- **Linux**: Follow [installation instructions](https://www.mongodb.com/docs/manual/administration/install-on-linux/)
- **MacOS**: `brew install mongodb-community`

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

## Running the Application

1. Make sure MongoDB is running
2. Make sure Ollama is running with the llama3.2 model
3. Start the application:
   ```
   streamlit run app.py
   ```
4. Access the application at http://localhost:8501

## Configuration

You can configure the MongoDB connection using environment variables:
- Set `MONGODB_URI` to your MongoDB connection string (default: `mongodb://localhost:27017/`)

## Usage

1. Upload documents through the sidebar
2. For images, EasyOCR will automatically extract text (first run may take longer as models download)
3. Ask questions in the chat interface
4. Select specific documents for targeted queries
5. Create new chat sessions or browse previous conversations

## Troubleshooting

If OCR isn't working:
- Check the console for detailed error messages
- For GPU acceleration, verify your CUDA setup
- If you encounter memory issues, try using smaller images or disable GPU in the code
- The first run will download models and may be slow, subsequent runs will be faster 