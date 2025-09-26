from flask import Flask, request, jsonify, send_from_directory
from src.app import RAGApplication
import os

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
        n_results = data.get('n_results', 3)
        
        if not question:
            print("Error: No question provided in request")
            return jsonify({'error': 'Question is required'}), 400
        
        print(f"Processing query: {question}")
        result = rag_app.query(question, n_results)
        print(f"Query response: {result['answer'][:50]}...")
        return jsonify({
            'answer': result['answer'],
            'context': result['context'],
            'metadata': result['metadata']
        })
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
