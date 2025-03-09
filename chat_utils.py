import streamlit as st
import pymongo
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
import traceback
import pandas as pd
from ollama_python.endpoints import GenerateAPI
# Load environment variables for MongoDB connection
load_dotenv()

def get_mongodb_connection():
    """
    Establish connection to MongoDB.
    
    Returns:
        dict: MongoDB client, database, and collections or None if connection fails
    """
    try:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        print(f"Connecting to MongoDB at {uri}")
        
        client = pymongo.MongoClient(uri)
        # Verify connection
        client.admin.command('ping')
        print("MongoDB connection successful")
        
        # Set up database and collections
        db = client.chatbot
        chats = db.chats
        messages = db.messages
        
        return {
            "client": client,
            "db": db,
            "chats": chats,
            "messages": messages
        }
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        traceback.print_exc()
        return None

def create_new_chat():
    """Create a new chat session in MongoDB."""
    if st.session_state.mongodb is None:
        print("Cannot create chat: MongoDB connection not available")
        return None
    
    try:
        # Generate a timestamp-based title
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_title = f"Chat {timestamp}"
        
        # Insert the new chat document
        chat = {
            "title": chat_title,
            "created_at": datetime.now()
        }
        result = st.session_state.mongodb["chats"].insert_one(chat)
        chat_id = str(result.inserted_id)
        print(f"Created new chat with ID: {chat_id}")
        
        return chat_id
    except Exception as e:
        print(f"Error creating new chat: {e}")
        traceback.print_exc()
        return None

def save_message(chat_id, role, content):
    """
    Save a message to MongoDB.
    
    Args:
        chat_id: The ID of the chat session
        role: Message role (user or assistant)
        content: Message content
        
    Returns:
        bool: True if successful, False otherwise
    """
    if st.session_state.mongodb is None:
        print("Cannot save message: MongoDB connection not available")
        return False
    
    try:
        # Create the message document
        message = {
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        
        # Insert into messages collection
        result = st.session_state.mongodb["messages"].insert_one(message)
        if result.inserted_id:
            print(f"Saved {role} message to chat {chat_id}")
            return True
        return False
    except Exception as e:
        print(f"Error saving message: {e}")
        traceback.print_exc()
        return False

def get_chat_messages(chat_id):
    """
    Retrieve messages for a specific chat.
    
    Args:
        chat_id: The ID of the chat session
        
    Returns:
        list: Chat messages sorted by timestamp
    """
    if st.session_state.mongodb is None:
        print("Cannot get messages: MongoDB connection not available")
        return []
    
    try:
        messages = list(st.session_state.mongodb["messages"].find(
            {"chat_id": chat_id},
            sort=[("timestamp", pymongo.ASCENDING)]
        ))
        return messages
    except Exception as e:
        print(f"Error retrieving chat messages: {e}")
        traceback.print_exc()
        return []

def get_all_chats():
    """
    Retrieve all chat sessions.
    
    Returns:
        list: All chat sessions sorted by creation time (most recent first)
    """
    if st.session_state.mongodb is None:
        print("Cannot get chats: MongoDB connection not available")
        return []
    
    try:
        chats = list(st.session_state.mongodb["chats"].find(
            sort=[("created_at", pymongo.DESCENDING)]
        ))
        return chats
    except Exception as e:
        print(f"Error retrieving chats: {e}")
        traceback.print_exc()
        return []

def render_chat_ui():
    """
    Render the chat interface and handle chat interactions.
    
    Returns:
        bool: True if successful, False otherwise
    """
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
    
    return True

def process_user_message(client, prompt, selected_document, category_info=None, similar_questions=None):
    """
    Process a user message and generate a response using the LLM.
    
    Args:
        client: The Ollama client (either ollama-python GenerateAPI or custom client)
        prompt: The user's message
        selected_document: The selected document for retrieval
        category_info: Optional category information for the prompt
        similar_questions: Optional similar questions for display
        
    Returns:
        str: The assistant's response
    """
    from document_utils import retrieve_context
    import inspect
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        
        # Display category info if available
        if category_info:
            main_issue, sub_issue = category_info
            st.info(f"📊 Category: **{main_issue}** → **{sub_issue}**", icon="📊")
            
            # Show similar questions if available
            if similar_questions:
                with st.expander(f"Similar Questions in '{sub_issue}'"):
                    for i, q in enumerate(similar_questions, 1):
                        st.markdown(f"**{i}.** {q['content']}")
                        if 'timestamp' in q:
                            try:
                                timestamp = datetime.fromisoformat(q['timestamp'])
                                st.caption(f"Asked on: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                            except:
                                pass
    
    # Get context for the prompt, filtering by selected document if applicable
    retrieved_context = retrieve_context(
        query=prompt, 
        top_k=3, 
        filter_document=selected_document
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
        try:
            print(f"Sending request to Ollama with {len(ollama_messages)} messages")
            print(f"Client type: {type(client).__name__}")
            
            # Determine which method to use based on the client type
            if hasattr(client, 'generate_chat_completion'):
                # Using ollama-python GenerateAPI
                print("Using generate_chat_completion from ollama-python")
                stream = client.generate_chat_completion(
                    messages=ollama_messages,
                    stream=True
                )
                
                # Process the streaming response from ollama-python
                for chunk in stream:
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                        # Pydantic model response
                        response = chunk.message.content
                    else:
                        # Dictionary response
                        response = chunk.get('message', {}).get('content', '')
                        if not response and 'content' in chunk:
                            response = chunk['content']
                    
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
            
            elif hasattr(client, 'chat'):
                # Using custom OllamaClient
                print("Using chat method from custom client")
                stream = client.chat(
                    model='llama3.2',
                    messages=ollama_messages,
                    stream=True
                )
                
                # Process the streaming response from custom client
                for chunk in stream:
                    if isinstance(chunk, dict):
                        if 'message' in chunk and 'content' in chunk['message']:
                            response = chunk['message']['content']
                        elif 'response' in chunk:
                            response = chunk['response']
                        elif 'content' in chunk:
                            response = chunk['content']
                        else:
                            print(f"Unknown response format: {chunk}")
                            response = str(chunk)
                    else:
                        print(f"Unexpected response type: {type(chunk)}")
                        response = str(chunk)
                    
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
            
            else:
                # No compatible methods found
                error_msg = f"Client {type(client).__name__} has no compatible methods. Available methods: {dir(client)}"
                print(error_msg)
                message_placeholder.markdown(f"Error: {error_msg}")
                full_response = f"Error: No compatible LLM methods found."
            
            # Show final response without cursor
            message_placeholder.markdown(full_response)
            print(f"Generated response of length {len(full_response)}")
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            message_placeholder.markdown(f"Error generating response: {str(e)}")
            full_response = f"Error generating response: {str(e)}"
    
    # Save assistant response to MongoDB
    if st.session_state.current_chat_id is not None and st.session_state.mongodb is not None:
        save_message(st.session_state.current_chat_id, "assistant", full_response)
    
    return full_response 