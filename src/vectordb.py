import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(self, collection_name: str = "rag_collection"):
        self.client = chromadb.Client()
        self.embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_model
        )

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into smaller chunks using LangChain's RecursiveCharacterTextSplitter.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            keep_separator=True
        )
        chunks = splitter.split_text(text)
        return chunks

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Process documents and add them to the vector database.

        Args:
            documents: List of documents with 'content' and optional 'metadata'
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
                all_ids.append(str(uuid.uuid4()))

        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Find documents similar to the query.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary with search results
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
