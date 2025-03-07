import streamlit as st
from ollama import Client
from langchain_ollama import OllamaEmbeddings
from pdftext.extraction import plain_text_output
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
import uuid
from datetime import datetime
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import json
import easyocr
from PIL import Image
import io
import platform
import numpy as np
from knowledge_graph import KnowledgeGraph
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define get_available_documents at the very top of the file
def get_available_documents():
    """
    Get a list of available documents with their upload times.
    Returns a list of tuples (doc_name, upload_time)
    """
    try:
        if "vector_store" not in st.session_state:
            print("Vector store not initialized yet")
            return []
            
        all_docs = st.session_state.vector_store.get()
        if all_docs and 'metadatas' in all_docs and all_docs['metadatas']:
            doc_times = {}
            for metadata in all_docs['metadatas']:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    upload_time = metadata.get("upload_time", None)
                    if upload_time:
                        try:
                            upload_time = datetime.fromisoformat(upload_time)
                        except (ValueError, TypeError):
                            upload_time = st.session_state.document_upload_times.get(source, datetime.min)
                    else:
                        upload_time = st.session_state.document_upload_times.get(source, datetime.min)
                    if source not in doc_times or upload_time > doc_times[source]:
                        doc_times[source] = upload_time
            doc_list = [(doc, time) for doc, time in doc_times.items()]
            doc_list.sort(key=lambda x: x[1], reverse=True)
            if "document_upload_times" in st.session_state:
                st.session_state.document_upload_times.update(doc_times)
            print(f"get_available_documents: Found {len(doc_list)} documents")
            return doc_list
        print("No documents found in vector store")
        return []
    except Exception as e:
        print(f"Error in get_available_documents: {e}")
        import traceback
        traceback.print_exc()
        return []

# Initialize Ollama Client
client = Client(host='http://localhost:11434')

# Initialize EasyOCR reader (lazy loading - only created when needed)
@st.cache_resource
def get_ocr_reader():
    """
    Initialize and return an EasyOCR reader object.
    Uses cache_resource to avoid reloading the model on each rerun.
    
    Returns:
        easyocr.Reader: Initialized OCR reader
    """
    print("Initializing EasyOCR reader (this may take a moment the first time)...")
    # Initialize for English by default - can be expanded to support more languages
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if CUDA is available
    print("EasyOCR reader initialized successfully")
    return reader

# OCR Processing Function
def extract_text_from_image(image_file):
    """
    Extract text from an image using EasyOCR
    
    Args:
        image_file: Uploaded image file from Streamlit
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Get the EasyOCR reader from cache
        reader = get_ocr_reader()
        
        # Open image using PIL
        image = Image.open(image_file)
        
        # Convert PIL image to numpy array for EasyOCR
        image_np = np.array(image)
        
        # Extract text using EasyOCR
        print(f"Performing OCR on image: {image_file.name}")
        
        # Perform OCR
        results = reader.readtext(image_np)
        
        # Extract text from results
        extracted_text = "\n".join([text for _, text, _ in results])
        
        # Debug: Print first 200 chars of extracted text
        preview = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        print(f"OCR result preview: {preview}")
        
        return extracted_text
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        import traceback
        traceback.print_exc()
        return f"Error processing image: {str(e)}"

# MongoDB Connection
def get_mongodb_connection():
    """
    Establish connection to MongoDB and return database instance.
    Uses environment variables for connection string or defaults to localhost.
    """
    try:
        # Try to get connection string from environment variable or use default
        mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
        mongo_client = MongoClient(mongo_uri)
        
        # Use 'chatbot_db' database
        db = mongo_client.chatbot_db
        
        # Create or access the collections
        chats_collection = db.chats
        messages_collection = db.messages
        
        # Create indexes for faster queries
        messages_collection.create_index([("chat_id", pymongo.ASCENDING)])
        messages_collection.create_index([("timestamp", pymongo.DESCENDING)])
        
        print("Successfully connected to MongoDB")
        return db
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        # Return None to indicate connection failure
        return None

# Initialize MongoDB connection
if "mongodb" not in st.session_state:
    st.session_state.mongodb = get_mongodb_connection()
    print(f"MongoDB connection established: {st.session_state.mongodb is not None}")

# Chat management functions
def create_new_chat():
    """Create a new chat session in MongoDB and return its ID"""
    if st.session_state.mongodb is None:
        print("Cannot create chat: MongoDB connection is None")
        return None
    
    try:
        chat_doc = {
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "title": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        }
        print(f"Attempting to create new chat with document: {chat_doc}")
        result = st.session_state.mongodb.chats.insert_one(chat_doc)
        chat_id = str(result.inserted_id)
        print(f"Created new chat with ID: {chat_id}")
        return chat_id
    except Exception as e:
        print(f"Error creating new chat: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_message(chat_id, role, content):
    """Save a message to MongoDB"""
    if st.session_state.mongodb is None or chat_id is None:
        print("Cannot save message: MongoDB connection or chat_id is None")
        return False
    
    try:
        message_doc = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        print(f"Saving message: {role} to chat {chat_id}")
        st.session_state.mongodb.messages.insert_one(message_doc)
        
        # Update the chat's updated_at timestamp
        print(f"Updating chat timestamp for chat_id: {chat_id}")
        update_result = st.session_state.mongodb.chats.update_one(
            {"_id": ObjectId(chat_id)},
            {"$set": {"updated_at": datetime.now()}}
        )
        print(f"Update result: matched={update_result.matched_count}, modified={update_result.modified_count}")
        return True
    except Exception as e:
        print(f"Error saving message: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_chat_messages(chat_id):
    """Retrieve all messages for a specific chat from MongoDB"""
    if st.session_state.mongodb is None or chat_id is None:
        print("Cannot get messages: MongoDB connection or chat_id is None")
        return []
    
    try:
        messages = list(st.session_state.mongodb.messages.find(
            {"chat_id": chat_id},
            {"_id": 0, "role": 1, "content": 1}
        ).sort("timestamp", pymongo.ASCENDING))
        return messages
    except Exception as e:
        print(f"Error retrieving messages: {e}")
        return []

def get_all_chats():
    """Retrieve all chat sessions from MongoDB"""
    if st.session_state.mongodb is None:
        print("Cannot get chats: MongoDB connection is None")
        return []
    
    try:
        chats = list(st.session_state.mongodb.chats.find().sort("updated_at", pymongo.DESCENDING))
        return chats
    except Exception as e:
        print(f"Error retrieving chats: {e}")
        return []

st.title("Chatbot with RAG & Document Upload")

# Create Chroma persist directory if it doesn't exist
PERSIST_DIRECTORY = "chroma_db"
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)
    print(f"Created persist directory at {PERSIST_DIRECTORY}")

# Initialize Embeddings
embed = OllamaEmbeddings(model="all-minilm")
print(f"Initialized embeddings with model: all-minilm")

# Initialize or load the Chroma vector store with proper configuration
if "vector_store" not in st.session_state:
    try:
        # Check if the Chroma DB already exists and load it
        st.session_state.vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embed,
            collection_name="document_collection"
        )
        print(f"Loaded existing Chroma vector store from {PERSIST_DIRECTORY}")
    except Exception as e:
        print(f"Error loading Chroma vector store: {e}")
        # Create a new Chroma DB if loading fails
        st.session_state.vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embed,
            collection_name="document_collection"
        )
        print(f"Created new Chroma vector store at {PERSIST_DIRECTORY}")

# Initialize session state for document upload times if it doesn't exist
if "document_upload_times" not in st.session_state:
    st.session_state.document_upload_times = {}

# Initialize Neo4j knowledge graph
if "knowledge_graph" not in st.session_state:
    try:
        print("Initializing knowledge graph...")
        st.session_state.knowledge_graph = KnowledgeGraph()
        print("Knowledge graph instance created")
    except Exception as e:
        print(f"Error initializing knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        # Important: Always set the variable, even on error
        st.session_state.knowledge_graph = None

# Check Neo4j connection if we have a knowledge graph instance
if "knowledge_graph" in st.session_state and st.session_state.knowledge_graph is not None:
    try:
        # Verify connection is still active
        connection_active = st.session_state.knowledge_graph.check_connection()
        print(f"Neo4j connection check: {'Active' if connection_active else 'Inactive'}")
    except Exception as e:
        print(f"Error checking Neo4j connection: {e}")
        # Don't set to None, as we might have partial functionality

# Replace the three tabs with just two tabs for Chat and Documents
sidebar_tab1, sidebar_tab2 = st.sidebar.tabs(["Chat", "Documents"])

with sidebar_tab1:
    st.header("Chat Sessions")
    
    # Initialize chat session ID (but don't create in MongoDB yet)
    if "current_chat_id" not in st.session_state:
        # Set to None initially - will create when user sends a message
        st.session_state.current_chat_id = None
        print("Initialized current_chat_id to None - will create actual chat when user sends a message")
    
    # Option to create a new chat session
    if st.button("New Chat"):
        # Clear the current chat ID to start fresh - actual MongoDB chat will be created on first message
        st.session_state.current_chat_id = None
        print("User requested new chat - cleared current_chat_id")
        st.rerun()
    
    # Option to load existing chat sessions
    chats = get_all_chats()
    if chats:
        chat_options = {str(chat["_id"]): chat["title"] for chat in chats}
        selected_chat = st.selectbox(
            "Load existing chat:",
            options=list(chat_options.keys()),
            format_func=lambda x: chat_options[x],
            index=0 if st.session_state.current_chat_id in chat_options else None
        )
        
        if selected_chat != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_chat
            print(f"User selected existing chat: {selected_chat}")
            st.rerun()
    
    # Keep basic category statistics but simplify
    if st.session_state.knowledge_graph is not None:
        try:
            with st.expander("Question Categories", expanded=False):
                stats = st.session_state.knowledge_graph.get_category_statistics()
                
                # Display overall distribution
                if stats:
                    categories = []
                    counts = []
                    
                    for issue, data in stats.items():
                        categories.append(issue)
                        counts.append(data['total'])
                    
                    if categories and counts:
                        # Create a proper DataFrame for the bar chart
                        chart_data = pd.DataFrame({"Count": counts}, index=categories)
                        st.bar_chart(chart_data)
                
                # Display detailed breakdown
                for issue, data in stats.items():
                    st.write(f"**{issue}** ({data['total']} questions)")
                    for subissue, count in data['subissues'].items():
                        st.write(f"  - {subissue}: {count}")
        except Exception as e:
            print(f"Error getting category statistics: {e}")
            import traceback
            traceback.print_exc()

with sidebar_tab2:
    st.header("Document Management")
    
    # File uploader for PDFs and text files
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload document files (PDF, TXT, or images)", 
        type=["pdf", "txt", "jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write({
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": uploaded_file.size
            })
            
            # Extract text content from uploaded file
            if uploaded_file.type == "application/pdf":
                content = plain_text_output(uploaded_file)
                print(f"Extracted text from PDF: {uploaded_file.name}")
            elif uploaded_file.type.startswith("image/"):
                # Use OCR for image files
                with st.status("Performing OCR on image (this may take a moment)..."):
                    content = extract_text_from_image(uploaded_file)
                    if content.startswith("Error"):
                        st.error(content)
                        content = ""
                    else:
                        st.success("OCR completed successfully!")
                        # Show preview of extracted text
                        with st.expander("Preview extracted text"):
                            st.write(content[:500] + "..." if len(content) > 500 else content)
                print(f"Extracted text from image using EasyOCR: {uploaded_file.name}")
            else:
                # Assume it's a text file
                file_bytes = uploaded_file.read()
                try:
                    content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = file_bytes.decode("latin1")
                print(f"Extracted text from TXT: {uploaded_file.name}")
        
            # Only process if we have content
            if content:
                # Split text into chunks for processing
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(content)
                print(f"Split document into {len(chunks)} chunks")
                
                if chunks:
                    # Create langchain Documents with metadata
                    documents = []
                    for i, chunk in enumerate(chunks):
                        doc_id = f"{uploaded_file.name}_{i}_{uuid.uuid4().hex[:8]}"
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "source": uploaded_file.name,
                                    "chunk_id": i,
                                    "document_id": doc_id,
                                    "upload_time": datetime.now().isoformat(),
                                    "file_type": uploaded_file.type
                                }
                            )
                        )
                    
                    # Add documents to the vector store
                    with st.status("Processing document..."):
                        st.session_state.vector_store.add_documents(documents)
                        # Explicitly persist after adding documents
                        st.session_state.vector_store.persist()
                        # Record upload time for sorting
                        st.session_state.document_upload_times[uploaded_file.name] = datetime.now()
                        print(f"Added and persisted {len(documents)} documents to Chroma")
                    
                    st.success(f"Document processed and indexed: {len(chunks)} chunks")
                else:
                    st.warning("No text content could be extracted from the document.")
            else:
                st.warning("No content could be extracted from the file.")
    
    # Get available documents
    available_documents = get_available_documents()
    # Add "None" as first option, then "All Documents", then individual documents
    doc_names = ["None", "All Documents"] + [doc[0] for doc in available_documents]
    
    # Document Selection for Querying
    st.subheader("Select Document to Query")
    selected_document = st.selectbox(
        "Filter queries to specific document:",
        options=doc_names,
        index=0,  # Default to "None" (direct LLM query)
        help="Select 'None' to query the LLM directly without document context, 'All Documents' to search across everything, or a specific document"
    )
    
    if selected_document == "None":
        st.info("📝 LLM Direct Mode: Queries will be sent directly to the language model without document context.")
    elif selected_document != "All Documents":
        # Show upload time for the selected document
        selected_doc_time = next((doc[1] for doc in available_documents if doc[0] == selected_document), None)
        if selected_doc_time and selected_document != "All Documents":
            st.info(f"Queries will be limited to: {selected_document}\nUploaded: {selected_doc_time.strftime('%Y-%m-%d %H:%M:%S')}")
        elif selected_document != "All Documents":
            st.info(f"Queries will be limited to: {selected_document}")

    # Sidebar: Document Deletion
    st.subheader("Delete Document")
    try:
        if available_documents:
            # Only show document names in the selection (not upload times)
            doc_sources = [doc[0] for doc in available_documents]
            
            if doc_sources:
                selected_doc_to_delete = st.selectbox(
                    "Select Document to Delete", 
                    options=doc_sources,
                    key="delete_document_selector"  # Unique key to avoid conflict with query selector
                )
                # Show upload time for the document to be deleted
                delete_doc_time = next((doc[1] for doc in available_documents if doc[0] == selected_doc_to_delete), None)
                if delete_doc_time:
                    st.caption(f"Uploaded: {delete_doc_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                if st.button("Delete Selected Document"):
                    # Get all document data
                    all_docs = st.session_state.vector_store.get()
                    
                    # Find all IDs associated with the selected document
                    ids_to_delete = [
                        doc_id for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']) 
                        if metadata.get("source") == selected_doc_to_delete
                    ]
                    
                    if ids_to_delete:
                        # Delete documents from vector store
                        st.session_state.vector_store.delete(ids=ids_to_delete)
                        # Remove from upload times tracking
                        if selected_doc_to_delete in st.session_state.document_upload_times:
                            del st.session_state.document_upload_times[selected_doc_to_delete]
                        print(f"Deleted {len(ids_to_delete)} chunks from document: {selected_doc_to_delete}")
                        st.success(f"Deleted document: {selected_doc_to_delete} ({len(ids_to_delete)} chunks)")
                        # Refresh the UI
                        st.rerun()
                    else:
                        st.warning("No chunks found for this document.")
        else:
            st.info("No documents available for deletion.")
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        print(f"Error in document deletion section: {e}")
        import traceback
        traceback.print_exc()

# Remove the visualization tab (sidebar_tab3) entirely

# Remove the code for showing the knowledge graph visualization page
# This section checks if we should show the knowledge graph visualization
if "show_knowledge_graph" in st.session_state:
    # Remove this flag since we don't need it anymore
    del st.session_state.show_knowledge_graph

# Main Chat section - existing code for displaying messages
# Display chat messages from MongoDB
if st.session_state.current_chat_id is not None and st.session_state.mongodb is not None:
    mongo_messages = get_chat_messages(st.session_state.current_chat_id)
    
    # Display previous chat messages
    for message in mongo_messages:
        # Only display the user's original query, not the combined prompt
        display_content = message["content"]
        if message["role"] == "user" and "Context:" in display_content and "User Query:" in display_content:
            # Extract only the user query part
            parts = display_content.split("User Query:")
            if len(parts) > 1:
                display_content = parts[1].strip()
        
        with st.chat_message(message["role"]):
            st.markdown(display_content)
elif st.session_state.current_chat_id is None and st.session_state.mongodb is not None:
    st.info("Start a new conversation or select an existing chat.")
elif st.session_state.mongodb is None:
    st.warning("MongoDB connection failed. Chat history will not be saved or retrieved.")

def retrieve_context(query, top_k=3, filter_document=None):
    """
    Retrieve context from the vector store based on similarity to the query.
    
    Args:
        query (str): The query text for context retrieval
        top_k (int): Number of documents to retrieve
        filter_document (str, optional): Document name to filter results by
        
    Returns:
        str: Combined context from retrieved documents or None if direct LLM mode
    """
    # If "None" is selected, bypass retrieval completely
    if filter_document == "None":
        print("Direct LLM mode selected: bypassing document retrieval")
        return None
        
    print(f"Retrieving context for query: {query}")
    try:
        # Create search filter if a specific document is selected
        filter_dict = None
        if filter_document and filter_document != "All Documents":
            filter_dict = {"source": filter_document}
            print(f"Filtering results to document: {filter_document}")
        
        # Create retriever from vector store for more efficient retrieval
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k,
                "filter": filter_dict
            }
        )
        
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # If no documents found with filter, try without filter as fallback
        if not retrieved_docs and filter_dict:
            print(f"No results found in {filter_document}. Searching all documents.")
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            retrieved_docs = retriever.get_relevant_documents(query)
        
        # Extract and format context from retrieved documents
        contexts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "Unknown")
            contexts.append(f"[Document: {source}, Chunk: {chunk_id}]\n{doc.page_content}")
            print(f"Retrieved document {i+1}: {source}, Chunk: {chunk_id}")
            
        if contexts:
            context = "\n\n".join(contexts)
            return context
        else:
            return "No relevant context found in the documents."
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return "Error retrieving context. Please try again."

# Connection Status in Sidebar
st.sidebar.header("System Status")
col1, col2, col3 = st.sidebar.columns(3)

# MongoDB Status
mongodb_status = "✅ Online" if st.session_state.mongodb is not None else "❌ Offline"
with col1:
    st.write("MongoDB")
    st.write(mongodb_status)

# Neo4j Status
neo4j_status = "✅ Online" if st.session_state.knowledge_graph is not None else "❌ Offline"
with col2:
    st.write("Neo4j")
    st.write(neo4j_status)

# Ollama Status
ollama_status = "✅ Online" if "client" in locals() else "❌ Offline"
with col3:
    st.write("Ollama")
    st.write(ollama_status)

if prompt := st.chat_input("Ask Me Questions?"):
    # Generate a unique ID for this question
    question_id = str(uuid.uuid4())
    
    # Category display variables
    category_info = None
    similar_questions = []  # Initialize as empty list
    
    # Add question to knowledge graph if available
    if st.session_state.knowledge_graph is not None and hasattr(st.session_state.knowledge_graph, 'connected') and st.session_state.knowledge_graph.connected:
        try:
            # Get the sub-issue category for the question
            sub_issue, main_issue = st.session_state.knowledge_graph.classify_message(prompt)
            
            # Add to knowledge graph
            st.session_state.knowledge_graph.add_question(
                question_id=question_id,
                content=prompt,
                chat_id=st.session_state.current_chat_id
            )
            
            # Store category info for display after user message
            category_info = (main_issue, sub_issue)
            
            # Get similar questions from the same category
            similar_questions = st.session_state.knowledge_graph.get_similar_questions(sub_issue)
        except Exception as e:
            print(f"Error processing question with knowledge graph: {e}")
            import traceback
            traceback.print_exc()
    
    # Get context for the prompt, filtering by selected document if applicable
    retrieved_context = retrieve_context(
        query=prompt, 
        top_k=3, 
        filter_document=selected_document if selected_document != "All Documents" else None
    )
    
    # Show retrieved context in an expander if available
    if retrieved_context:
        with st.expander("Retrieved Context"):
            st.markdown(retrieved_context)
    elif selected_document == "None":
        st.info("Direct LLM mode: No document context used.")
    
    # Create a new MongoDB chat if this is the first message and connection exists
    if st.session_state.current_chat_id is None and st.session_state.mongodb is not None:
        st.session_state.current_chat_id = create_new_chat()
        print(f"Created new chat session with ID: {st.session_state.current_chat_id}")
    
    # Create combined prompt with retrieved context if available
    if retrieved_context:
        combined_prompt = f"Context: {retrieved_context}\n\nUser Query: {prompt}"
    else:
        combined_prompt = prompt
    
    # Add user message to MongoDB
    if st.session_state.current_chat_id is not None and st.session_state.mongodb is not None:
        success = save_message(st.session_state.current_chat_id, "user", combined_prompt)
        if not success:
            st.warning("Failed to save message to database")
    
    # Display only the query part (not the context) to the user
    with st.chat_message("user"):
        st.markdown(prompt)
        
        # Display category info if available
        if category_info:
            main_issue, sub_issue = category_info
            st.info(f"📊 Category: **{main_issue}** → **{sub_issue}**", icon="📊")
            
            # Show similar questions if available
            if 'similar_questions' in locals() and similar_questions:
                with st.expander(f"Similar Questions in '{sub_issue}'"):
                    for i, q in enumerate(similar_questions, 1):
                        st.markdown(f"**{i}.** {q['content']}")
                        if 'timestamp' in q:
                            try:
                                timestamp = datetime.fromisoformat(q['timestamp'])
                                st.caption(f"Asked on: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            except:
                                pass
    
    # Process with Ollama and display streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Get messages from MongoDB if connection exists
        if st.session_state.current_chat_id is not None and st.session_state.mongodb is not None:
            mongo_messages = get_chat_messages(st.session_state.current_chat_id)
            ollama_messages = [{"role": msg["role"], "content": msg["content"]} for msg in mongo_messages]
        else:
            # Fallback to just the current message if MongoDB is not available
            ollama_messages = [{"role": "user", "content": combined_prompt}]
        
        # Stream the response
        stream = client.chat(model='llama3.2', messages=ollama_messages, stream=True)
        for chunk in stream:
            response = chunk['message']['content']
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    # Save assistant response to MongoDB
    if st.session_state.current_chat_id is not None and st.session_state.mongodb is not None:
        save_message(st.session_state.current_chat_id, "assistant", full_response)
