from flask import Flask, request, jsonify, send_from_directory
from src.app import RAGApplication, DatabaseChatMemory
import os
import uuid

app = Flask(__name__, template_folder='templates')

try:
    print("Initializing RAGApplication...")
    rag_app = RAGApplication()
    print("RAGApplication initialized successfully")
except Exception as e:
    print(f"Failed to initialize RAGApplication: {str(e)}")
    raise

@app.route('/')
def index():
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
    try:
        data = request.get_json()
        question = data.get('question')
        user_id = data.get('user_id', 'default_user')
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        if not question:
            print("Error: No question provided in request")
            return jsonify({'error': 'Question is required'}), 400
        
        print(f"Processing query: {question} for user {user_id}, session {session_id}")
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
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default_user')
        session_id = data.get('session_id')
        
        if not session_id:
            print("Error: No session_id provided")
            return jsonify({'error': 'Session ID is required'}), 400
        
        memory = DatabaseChatMemory(user_id, session_id, rag_app.db_path)
        history = memory.load_conversation()
        print(f"Retrieved history for user {user_id}, session {session_id}")
        return jsonify({'history': history})
    except Exception as e:
        print(f"Error retrieving history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export_knowledge', methods=['GET'])
def export_knowledge():
    try:
        output_file = "knowledge_base.json"
        rag_app.export_knowledge_base(output_file)
        return jsonify({'message': f'Knowledge base exported to {output_file}'})
    except Exception as e:
        print(f"Error exporting knowledge base: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
