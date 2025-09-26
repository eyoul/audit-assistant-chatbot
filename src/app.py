import os
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from src.vectordb import VectorDB
from dotenv import load_dotenv

load_dotenv()

class RAGApplication:
    def __init__(self):
        try:
            print("Initializing VectorDB...")
            self.vector_db = VectorDB()
            print("Loading documents...")
            self.documents = self.load_documents()
            print(f"Loaded {len(self.documents)} documents")
            self.vector_db.add_documents(self.documents)
            
            print("Initializing Groq LLM...")
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in .env")
            
            self.llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=4096
            )
            
            print("Setting up prompt template...")
            self.prompt_template = ChatPromptTemplate.from_template(
                """
                You are a helpful AI assistant. Use the following context to answer the question accurately. If the context doesn't have the information, say "I don't have enough information from the documents."
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
            )
            
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            print("RAGApplication initialized successfully")
        except Exception as e:
            print(f"Error initializing RAGApplication: {str(e)}")
            raise

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load documents from the data/ directory, supporting PDF and text files.
        """
        data_dir = "data/"
        results = []
        
        if not os.path.exists(data_dir):
            print(f"Data directory {data_dir} does not exist.")
            return results

        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            try:
                if filename.endswith('.pdf'):
                    with open(filepath, 'rb') as file:
                        pdf = PdfReader(file)
                        content = ""
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                content += text + "\n"
                    doc = {
                        'content': content.strip(),
                        'metadata': {'filename': filename, 'type': 'pdf'}
                    }
                    results.append(doc)
                elif filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        content = file.read()
                    doc = {
                        'content': content.strip(),
                        'metadata': {'filename': filename, 'type': 'txt'}
                    }
                    results.append(doc)
                else:
                    print(f"Unsupported file type: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        return results

    def query(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Answer questions using retrieved context.

        Args:
            question: User's question
            n_results: Number of context chunks to retrieve

        Returns:
            Dictionary with answer and context information
        """
        try:
            search_results = self.vector_db.search(question, n_results)
            context = "\n\n".join(search_results['documents'])
            response = self.chain.invoke({
                'context': context,
                'question': question
            })
            return {
                'answer': response['text'],
                'context': search_results['documents'],
                'metadata': [m['filename'] for m in search_results['metadatas']]
            }
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise
