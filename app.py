import streamlit as st
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv
import pandas as pd
import json

# Import our modular utilities
from document_utils import initialize_vector_store, render_document_management_ui
from chat_utils import get_mongodb_connection, render_chat_ui, process_user_message
from knowledge_graph import KnowledgeGraph

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Set app title
st.title("RAG Chatbot with Document Processing")

# Initialize session state values
if "document_upload_times" not in st.session_state:
    st.session_state.document_upload_times = {}

# Initialize Ollama Client
try:
    # Try using direct Ollama API first (most reliable)
    import requests
    
    # Create a simple client that mimics the ollama-python API but uses direct HTTP requests
    class OllamaClient:
        def __init__(self, base_url="http://localhost:11434"):
            self.base_url = base_url
            
        def chat(self, model, messages, stream=False):
            url = f"{self.base_url}/api/chat"
            payload = {"model": model, "messages": messages, "stream": stream}
            
            if not stream:
                response = requests.post(url, json=payload)
                return response.json()
            else:
                # Streaming implementation
                response = requests.post(url, json=payload, stream=True)
                for line in response.iter_lines():
                    if line:
                        try:
                            yield json.loads(line)
                        except:
                            yield {"message": {"content": line.decode('utf-8')}}
    
    # Test if Ollama is running
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        client = OllamaClient()
        print("Direct Ollama client created and connected successfully")
    else:
        raise Exception(f"Ollama API not responding: {response.status_code}")
        
except Exception as e:
    print(f"Direct API connection failed: {e}, trying ollama-python...")
    
    try:
        # Try to import from ollama-python library as fallback
        from ollama_python.endpoints import GenerateAPI
        
        # Create Ollama client
        client = GenerateAPI(base_url="http://localhost:11434")
        print("Ollama client initialized successfully using ollama-python library")
        
    except Exception as e:
        print(f"Error initializing any Ollama client: {e}")
        import traceback
        traceback.print_exc()
        st.error("Failed to initialize Ollama client. Please make sure Ollama is running.")
        client = None

# Initialize MongoDB connection
if "mongodb" not in st.session_state:
    st.session_state.mongodb = get_mongodb_connection()
    if st.session_state.mongodb is None:
        st.warning("Failed to connect to MongoDB. Chat history will not be saved or retrieved.")

# Initialize Chroma vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = initialize_vector_store()

# Initialize Neo4j knowledge graph
if "knowledge_graph" not in st.session_state:
    try:
        print("Initializing knowledge graph...")
        st.session_state.knowledge_graph = KnowledgeGraph()
        print("Knowledge graph instance created")
        
        # Verify connection is still active
        connection_active = st.session_state.knowledge_graph.check_connection()
        print(f"Neo4j connection check: {'Active' if connection_active else 'Inactive'}")
    except Exception as e:
        print(f"Error initializing knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        # Important: Always set the variable, even on error
        st.session_state.knowledge_graph = None

# Create sidebar tabs for different functionality
sidebar_tab1, sidebar_tab2 = st.sidebar.tabs(["Chat", "Documents"])

# Chat tab
with sidebar_tab1:
    render_chat_ui()
    
    # Show basic category statistics
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

# Documents tab
with sidebar_tab2:
    selected_document = render_document_management_ui()

# Chat input and processing
if prompt := st.chat_input("Ask Me Questions?"):
    # Generate a unique ID for this question
    question_id = str(uuid.uuid4())
    
    # Category display variables
    category_info = None
    similar_questions = []  # Initialize as empty list
    
    # Add question to knowledge graph if available
    if (st.session_state.knowledge_graph is not None and 
        hasattr(st.session_state.knowledge_graph, 'connected') and 
        st.session_state.knowledge_graph.connected):
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
    
    # Process the user message
    if client:
        process_user_message(
            client=client,
            prompt=prompt,
            selected_document=selected_document,
            category_info=category_info,
            similar_questions=similar_questions
        )
    else:
        st.error("Ollama client is not available. Please make sure Ollama is running.")
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown("Sorry, I can't process your request at the moment because the LLM is not available.")
