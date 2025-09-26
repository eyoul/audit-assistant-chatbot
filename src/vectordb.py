import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(self, collection_name: str = "rag_collection", chunk_size: int = 500, chunk_overlap: int = 50):
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

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
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
