import streamlit as st
import os
from datetime import datetime
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import pandas as pd
import traceback

# Path for Chroma persistence
PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")

def initialize_vector_store():
    """Initialize Chroma vector store with persistence."""
    try:
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=None)
        print(f"Loaded existing Chroma vector store from {PERSIST_DIRECTORY}")
        return vector_store
    except Exception as e:
        print(f"Error loading Chroma vector store: {e}")
        # Create a new vector store if loading fails
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=None)
        vector_store.persist()
        print(f"Created new Chroma vector store at {PERSIST_DIRECTORY}")
        return vector_store

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
        traceback.print_exc()
        return []

def process_text_document(content, file_name, file_type):
    """Process text content into chunks and add to vector store."""
    # Split text into chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    print(f"Split document into {len(chunks)} chunks")
    
    if not chunks:
        return False, "No text chunks could be extracted."
    
    # Create langchain Documents with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        doc_id = f"{file_name}_{i}_{uuid.uuid4().hex[:8]}"
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "chunk_id": i,
                    "document_id": doc_id,
                    "upload_time": datetime.now().isoformat(),
                    "file_type": file_type
                }
            )
        )
    
    # Add documents to the vector store
    st.session_state.vector_store.add_documents(documents)
    # Explicitly persist after adding documents
    st.session_state.vector_store.persist()
    # Record upload time for sorting
    st.session_state.document_upload_times[file_name] = datetime.now()
    print(f"Added and persisted {len(documents)} documents to Chroma")
    
    return True, f"Document processed and indexed: {len(chunks)} chunks"

def delete_document(document_name):
    """Delete a document from the vector store by its name."""
    try:
        # Get all document data
        all_docs = st.session_state.vector_store.get()
        
        # Find all IDs associated with the document
        ids_to_delete = [
            doc_id for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas']) 
            if metadata.get("source") == document_name
        ]
        
        if not ids_to_delete:
            return False, "No chunks found for this document."
            
        # Delete documents from vector store
        st.session_state.vector_store.delete(ids=ids_to_delete)
        # Remove from upload times tracking
        if document_name in st.session_state.document_upload_times:
            del st.session_state.document_upload_times[document_name]
        
        print(f"Deleted {len(ids_to_delete)} chunks from document: {document_name}")
        return True, f"Deleted document: {document_name} ({len(ids_to_delete)} chunks)"
    except Exception as e:
        print(f"Error deleting document: {e}")
        traceback.print_exc()
        return False, f"Error: {str(e)}"

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
            return None
    except Exception as e:
        print(f"Error retrieving context: {e}")
        traceback.print_exc()
        return None

def render_document_management_ui():
    """Render the document management UI components."""
    st.header("Document Management")
    
    # File uploader for documents
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload document files (PDF, TXT, or images)", 
        type=["pdf", "txt", "jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        from ocr_utils import extract_text_from_image
        from pdf_utils import plain_text_output
        
        for uploaded_file in uploaded_files:
            st.write({
                "filename": uploaded_file.name,
                "filetype": uploaded_file.type,
                "filesize": uploaded_file.size
            })
            
            # Extract text content from uploaded file
            content = ""
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
                print(f"Extracted text from image using OCR: {uploaded_file.name}")
            else:
                # Assume it's a text file
                file_bytes = uploaded_file.read()
                try:
                    content = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content = file_bytes.decode("latin1")
                print(f"Extracted text from TXT: {uploaded_file.name}")
        
            # Process the content if it's available
            if content:
                with st.status("Processing document..."):
                    success, message = process_text_document(
                        content, 
                        uploaded_file.name, 
                        uploaded_file.type
                    )
                
                if success:
                    st.success(message)
                else:
                    st.warning(message)
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
        if selected_doc_time:
            st.info(f"Queries will be limited to: {selected_document}\nUploaded: {selected_doc_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info(f"Queries will be limited to: {selected_document}")

    # Document Deletion section
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
                    success, message = delete_document(selected_doc_to_delete)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.warning(message)
        else:
            st.info("No documents available for deletion.")
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        print(f"Error in document deletion section: {e}")
        traceback.print_exc()
        
    return selected_document 