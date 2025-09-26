import os
import json
import sqlite3
import yaml
from typing import List, Dict, Any
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from src.vectordb import VectorDB
from dotenv import load_dotenv

load_dotenv()

class DatabaseChatMemory:
    def __init__(self, user_id, session_id, db_path):
        self.user_id = user_id
        self.session_id = session_id
        self.db_path = db_path

    def save_message(self, role: str, content: str):
        """Save a single message to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO chat_messages (user_id, session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (self.user_id, self.session_id, role, content, datetime.now()))
            conn.commit()

    def load_conversation(self) -> list:
        """Load the full conversation in order."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT role, content FROM chat_messages
                WHERE user_id = ? AND session_id = ?
                ORDER BY timestamp ASC
            """, (self.user_id, self.session_id))
            return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

class RAGApplication:
    def __init__(self, db_path="chat_history.db", config_path="config.yaml"):
        try:
            print("Loading configuration...")
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            print("Initializing SQLite database...")
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp DATETIME
                    )
                """)
                conn.commit()
            
            self.db_path = db_path
            print("Initializing VectorDB...")
            self.vector_db = VectorDB(
                collection_name=self.config['vector_db']['collection_name'],
                chunk_size=self.config['vector_db']['chunk_size'],
                chunk_overlap=self.config['vector_db']['chunk_overlap']
            )
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
                model=self.config['model']['name'],
                temperature=self.config['model']['temperature'],
                max_tokens=self.config['model']['max_tokens']
            )
            
            print("Setting up prompt template...")
            self.prompt_template = ChatPromptTemplate.from_template(
                self.config['prompt']['system'] + """

                Context:
                {context}
                
                Conversation History:
                {history}
                
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

    def query(self, question: str, user_id: str, session_id: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Answer questions using retrieved context and conversation history.

        Args:
            question: User's question
            user_id: Unique user identifier
            session_id: Unique session identifier
            n_results: Number of context chunks to retrieve

        Returns:
            Dictionary with answer and context information
        """
        try:
            memory = DatabaseChatMemory(user_id, session_id, self.db_path)
            memory.save_message("user", question)
            history = memory.load_conversation()
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            search_results = self.vector_db.search(question, n_results)
            context = "\n\n".join(search_results['documents'])
            response = self.chain.invoke({
                'context': context,
                'history': history_text,
                'question': question
            })
            memory.save_message("assistant", response['text'])
            return {
                'answer': response['text'],
                'context': search_results['documents'],
                'metadata': [m['filename'] for m in search_results['metadatas']]
            }
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            raise

    def export_knowledge_base(self, output_file: str = "knowledge_base.json"):
        """
        Export knowledge base from PDFs and vector database to a JSON file.

        Args:
            output_file: Path to save the JSON file

        Returns:
            None
        """
        try:
            knowledge_base = {
                "pdfs": self.load_documents(),
                "vectordb": self.vector_db.export_all_documents()
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            print(f"Knowledge base exported to {output_file}")
        except Exception as e:
            print(f"Error exporting knowledge base: {str(e)}")
            raise
