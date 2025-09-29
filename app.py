"""
Flask Web Application for Audit Assistant Chatbot

This module provides a RESTful API for an AI-powered audit assistant chatbot.
It handles user queries, maintains conversation history, and provides RAG-based responses.
"""

from flask import Flask, request, jsonify, send_from_directory
from src.app import RAGApplication, DatabaseChatMemory
import os
import uuid

# Initialize Flask application with templates directory
app = Flask(__name__, template_folder='templates')

# Initialize the RAG (Retrieval-Augmented Generation) application
try:
    print("Initializing RAGApplication...")
    rag_app = RAGApplication()  # Initialize the core RAG functionality
    print("RAGApplication initialized successfully")
except Exception as e:
    print(f"Failed to initialize RAGApplication: {str(e)}")
    raise

@app.route('/')
def index():
    """
    Serve the main application interface.
    
    Returns:
        Response: The index.html file or an error message if not found.
    """
    try:
        if not os.path.exists('templates/index.html'):
            print("Error: index.html not found in templates/")
            return "Error: index.html not found", 500
        print("Serving index.html")
        return send_from_directory('templates', 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/api/query', methods=['POST'])
def query():
    """
    Handle user queries and return AI-generated responses.
    
    Expected JSON payload:
        - question (str): The user's query
        - user_id (str, optional): Unique user identifier
        - session_id (str, optional): Session identifier (new one generated if not provided)
        
    Returns:
        JSON: Response containing answer, context, metadata, and session_id
    """
    try:
        data = request.get_json()
        question = data.get('question')
        user_id = data.get('user_id', 'default_user')
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        if not question:
            print("Error: No question provided in request")
            return jsonify({'error': 'Question is required'}), 400
        
        print(f"Processing query: {question} for user {user_id}, session {session_id}")
        # Query the RAG model with the user's question
        result = rag_app.query(question, user_id, session_id, n_results=3)
        print(f"Query response: {result['answer'][:50]}...")
        
        return jsonify({
            'answer': result['answer'],
            'context': result['context'],
            'metadata': result['metadata'],
            'session_id': session_id
        })
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['POST'])
def get_history():
    """
    Retrieve conversation history for a specific user and session.
    
    Expected JSON payload:
        - user_id (str, optional): User identifier
        - session_id (str): Session identifier (required)
        
    Returns:
        JSON: Conversation history for the specified session
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        session_id = data.get('session_id')
        
        if not session_id:
            print("Error: No session_id provided")
            return jsonify({'error': 'Session ID is required'}), 400
        
        # Load conversation history from the database
        memory = DatabaseChatMemory(user_id, session_id, rag_app.db_path)
        history = memory.load_conversation()
        print(f"Retrieved history for user {user_id}, session {session_id}")
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error retrieving history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_knowledge', methods=['GET'])
def export_knowledge():
    """
    Export the current knowledge base to a JSON file.
    
    Returns:
        JSON: Success message or error details
    """
    try:
        output_file = "knowledge_base.json"
        rag_app.export_knowledge_base(output_file)
        return jsonify({
            'message': f'Knowledge base exported to {output_file}',
            'file_path': output_file
        })
    except Exception as e:
        print(f"Error exporting knowledge base: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    """
    Main entry point for the Flask application.
    Starts the development server on all available network interfaces.
    """
    print("Starting Flask server...")
    # Run the Flask development server
    # - debug=True enables auto-reload and debug mode
    # - host='0.0.0.0' makes the server accessible from other devices on the network
    # - port=5000 is the default Flask port
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        if e.errno == 10038:
            print("Error: Another instance of the server is already running.")
        else:
            print(f"Error starting Flask server: {str(e)}")
