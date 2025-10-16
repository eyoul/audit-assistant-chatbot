import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from .vectordb import VectorDB
from .app import RAGApplication
import json

# Load eval data
with open('data/eval_data.json', 'r') as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# Init your RAG components (use config)
db = VectorDB()  # Assumes it loads from Chroma
llm = RAGApplication()  # Your Groq LLM

# Run eval (retrieves contexts via your search, generates answers)
# ... (keep the existing imports at the top)

# Run eval (retrieves contexts via your search, generates answers)
# ... (keep the existing imports)

def generate_answers(dataset):
    results = []
    for row in dataset:
        # Get the response
        response = llm.query(
            question=row['question'],
            user_id="evaluation_user",
            session_id="eval_session_1"
        )
        
        # Get search results
        retrieved = db.search(row['question'], n_results=3)
        contexts = [
            doc if isinstance(doc, str) else getattr(doc, 'page_content', str(doc))
            for doc in retrieved
        ]
        
        # Format for RAGAS
        results.append({
            'question': row['question'],
            'answer': str(response),  # Ensure response is string
            'contexts': contexts,
            'ground_truths': [row.get('ground_truth', '')]  # RAGAS expects a list for ground_truths
        })
    return results

# Generate answers
results = generate_answers(dataset)

# Create a dictionary of lists for the dataset
data_dict = {
    'question': [r['question'] for r in results],
    'answer': [r['answer'] for r in results],
    'contexts': [r['contexts'] for r in results],
    'ground_truths': [r['ground_truths'] for r in results]
}

# Create the dataset
ragas_dataset = Dataset.from_dict(data_dict)

# Run evaluation
try:
    evaluation_results = evaluate(
        ragas_dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=llm
    )

    print("Evaluation Results:")
    print(evaluation_results)

    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
        
except Exception as e:
    print(f"Error during evaluation: {str(e)}")
    print("Dataset format:", {k: type(v) for k, v in data_dict.items()})
    print("Sample data:", {k: v[0] if v else None for k, v in data_dict.items()})
