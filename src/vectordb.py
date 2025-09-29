"""
Vector Database Module for RAG (Retrieval-Augmented Generation) System

This module provides a VectorDB class that handles document storage, embedding,
and similarity search using ChromaDB and Sentence Transformers.
"""

import uuid
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    """
    A vector database wrapper for document storage and retrieval using ChromaDB.
    
    This class handles document chunking, embedding, and similarity search operations.
    It uses SentenceTransformer for embeddings and provides methods for adding documents
    and performing semantic searches.
    
    Args:
        collection_name: Name of the Chroma collection to use or create
        chunk_size: Maximum size of text chunks (in characters)
        chunk_overlap: Overlap between consecutive chunks (in characters)
        persist_directory: Directory to persist the database (None for in-memory)
    """
    def __init__(self, collection_name: str = "rag_collection", 
                 chunk_size: int = 500, 
                 chunk_overlap: int = 50, 
                 persist_directory: Optional[str] = None):
        # Use on-disk Chroma if a persist directory is provided; otherwise use in-memory
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_model
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
        
        The splitting is done recursively by trying different separators in order:
        1. Double newlines (\n\n)
        2. Single newlines (\n)
        3. Periods followed by spaces
        4. Spaces
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List[str]: List of text chunks, each not exceeding chunk_size
            
        Example:
            >>> db = VectorDB()
            >>> chunks = db.chunk_text("This is a test.\n\nThis is another paragraph.")
            >>> len(chunks) > 0
            True
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=True
        )
        chunks = splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process documents and add them to the vector database.
        
        Each document should be a dictionary with at least a 'content' key.
        Optional 'metadata' can be included for additional context.
        
        Args:
            documents: List of document dictionaries. Each dictionary should contain:
                - content (str): The text content of the document
                - metadata (dict, optional): Additional metadata like filename, source, etc.
                
        Returns:
            None
            
        Example:
            >>> db = VectorDB()
            >>> docs = [
            ...     {"content": "This is a test document", "metadata": {"source": "test.txt"}},
            ...     {"content": "Another document", "metadata": {"source": "doc2.txt"}}
            ... ]
            >>> db.add_documents(docs)
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            content = doc['content']
            metadata = doc.get('metadata', {})
            chunks = self.chunk_text(content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append(metadata)
                # Deterministic ID from content + filename + index to avoid duplicates across restarts
                base = f"{metadata.get('filename','')}-{i}-{len(chunk)}-{hashlib.md5(chunk.encode('utf-8')).hexdigest()}"
                all_ids.append(base)

        if all_chunks:
            # Use upsert to avoid duplicate key errors and ensure idempotent ingestion
            if hasattr(self.collection, 'upsert'):
                self.collection.upsert(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids
                )
            else:
                self.collection.add(
                    documents=all_chunks,
                    metadatas=all_metadatas,
                    ids=all_ids
                )

    def is_empty(self) -> bool:
        """
        Check if the vector database collection is empty.
        
        Uses the most efficient method available to determine if the collection
        contains any documents.
        
        Returns:
            bool: True if the collection is empty, False otherwise
            
        Note:
            This method includes a fallback mechanism in case the direct count()
            method is not available in the ChromaDB version being used.
        """
        try:
            return self.collection.count() == 0
        except Exception:
            # Fallback: treat as not empty if count is unavailable
            results = self.collection.get(include=["ids"])  # may be large on huge sets
            return len(results.get('ids', [])) == 0

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Perform a semantic search to find documents similar to the query.
        
        The search uses cosine similarity on the vector embeddings of the query
        and documents to find the most relevant matches.
        
        Args:
            query: The search query string
            n_results: Maximum number of results to return (default: 5)
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - documents: List of matching document texts
                - metadatas: List of metadata dictionaries for each match
                - distances: List of cosine distances (lower is more similar)
                - ids: List of document IDs for the matches
                
        Example:
            >>> db = VectorDB()
            >>> results = db.search("test query", n_results=3)
            >>> len(results['documents']) <= 3
            True
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0],
            'ids': results['ids'][0]
        }

    def export_all_documents(self) -> List[Dict[str, Any]]:
        """
        Export all documents and metadata from the vector database.

        Returns:
            List of dictionaries containing document content and metadata
        """
        results = self.collection.get()
        return [
            {
                'content': doc,
                'metadata': meta,
                'id': id_
            }
            for doc, meta, id_ in zip(
                results['documents'],
                results['metadatas'],
                results['ids']
            )
        ]
