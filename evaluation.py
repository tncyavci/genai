import os
import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# Import your chatbot and related processors
# This part will likely need adjustment for headless operation
from chat_bot import PDFChatBot 
# from pdf_processor import PDFProcessor # If needed directly
# from text_processor import TextProcessor # If needed directly
# from vector_store import VectorStore # If needed directly

# --- Configuration ---
# Ensure OPENAI_API_KEY is set in your environment variables
# or configure RAGAs to use a different LLM
# For local LLM with RAGAs, it's more involved: https://docs.ragas.io/en/stable/how-tos/customisations/custom_llm.html

# Specify the GGUF model path if you intend to use it for your ChatBot
GGUF_MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf" 
# Or set to None if not using GGUF or if PDFChatBot handles this internally

# Path to the PDF document(s) to use for evaluation
EVAL_PDF_PATH = "path/to/your/evaluation_document.pdf" # REPLACE with your actual PDF path

# Define your evaluation questions
EVALUATION_QUESTIONS = [
    "What is the total revenue?",
    "Summarize the key financial risks.",
    "Who is the CEO of the company?",
    # Add more questions relevant to your EVAL_PDF_PATH
]

# Define ground truths for some metrics (optional but recommended)
# Ensure the order matches EVALUATION_QUESTIONS
GROUND_TRUTHS = [
    ["The total revenue is $X million."], # Replace with actual ground truth from your PDF
    ["The key financial risks include market volatility and operational failures."], # Replace
    ["The CEO is Jane Doe."], # Replace
    # Add corresponding ground truths
]

async def run_evaluation():
    """
    Runs the RAG evaluation using RAGAs.
    """
    print("Initializing ChatBot for evaluation...")
    
    # Initialize your PDFChatBot
    # This is a critical part that might need adjustment.
    # Ensure your PDFChatBot can be initialized without Streamlit and
    # that it loads the GGUF model correctly if GGUF_MODEL_PATH is set.
    try:
        # Option 1: If your PDFChatBot can take gguf_model_path directly
        chatbot = PDFChatBot(gguf_model_path=GGUF_MODEL_PATH if os.path.exists(GGUF_MODEL_PATH) else None)
        
        # Option 2: If it relies on environment variables or other config
        # os.environ["CUSTOM_MODEL_PATH"] = GGUF_MODEL_PATH # If applicable
        # chatbot = PDFChatBot(use_local_llm=True, model_choice="gguf_model") # Or your relevant choice

        llm_info = chatbot.llm_service.get_service_info()
        print(f"ChatBot Initialized. LLM Service: {llm_info}")
        if not llm_info or llm_info.get('status') != 'active':
             if not chatbot.llm_service.gguf_service or not chatbot.llm_service.gguf_service.llm:
                print(f"Warning: LLM service might not be fully active. GGUF loaded: {bool(chatbot.llm_service.gguf_service and chatbot.llm_service.gguf_service.llm)}")
                # Depending on your RAGAs LLM setup, this might be fine if RAGAs uses a different LLM.
    except Exception as e:
        print(f"Error initializing ChatBot: {e}")
        print("Please ensure your PDFChatBot can be initialized headlessly and check model paths.")
        return

    print(f"Processing evaluation document: {EVAL_PDF_PATH}...")
    if not os.path.exists(EVAL_PDF_PATH):
        print(f"Error: Evaluation PDF not found at {EVAL_PDF_PATH}")
        return

    # Simulate file upload for PDF processing
    # PDFChatBot.process_pdf_file expects a file-like object with a 'name' attribute
    class MockUploadedFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self.path = path
        def read(self):
            with open(self.path, 'rb') as f:
                return f.read()

    try:
        mock_file = MockUploadedFile(EVAL_PDF_PATH)
        processing_result = chatbot.process_pdf_file(mock_file)
        if "✅" not in processing_result:
            print(f"Error processing PDF: {processing_result}")
            return
        print(f"PDF processed: {processing_result}")
    except Exception as e:
        print(f"Error during PDF processing for evaluation: {e}")
        return

    print("Generating answers and collecting contexts for evaluation questions...")
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [] 
    }

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"  Processing question: {question}")
        try:
            # 1. Retrieve context using your chatbot's retrieval service
            # The n_results might need tuning
            context_result = chatbot.retrieval_service.retrieve_context(query=question, n_results=3)
            retrieved_contexts = [doc.page_content for doc in context_result.results]
            
            # 2. Generate answer using your chatbot's LLM
            # generate_single_response is preferred as it doesn't update UI history
            answer = chatbot.generate_single_response(query=question) 

            data["question"].append(question)
            data["answer"].append(answer if answer else "Error: No answer generated")
            data["contexts"].append(retrieved_contexts if retrieved_contexts else ["Error: No context retrieved"])
            if i < len(GROUND_TRUTHS): # Add ground truth if available
                 data["ground_truth"].append(GROUND_TRUTHS[i])
            else: # Add placeholder if not
                 data["ground_truth"].append(["No ground truth provided for this question."])


            print(f"    Contexts: {len(retrieved_contexts)} chunks")
            print(f"    Answer: {answer[:100]}...") # Print first 100 chars of answer
            
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            data["question"].append(question)
            data["answer"].append(f"Error: {e}")
            data["contexts"].append([])
            if i < len(GROUND_TRUTHS):
                 data["ground_truth"].append(GROUND_TRUTHS[i])
            else:
                 data["ground_truth"].append(["No ground truth provided for this question."])


    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(data)
    print("\nDataset for RAGAs prepared:")
    print(dataset)

    print("\nRunning RAGAs evaluation...")
    # Define metrics
    # Note: context_recall requires ground_truth
    # answer_similarity also benefits from ground_truth
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]
    if all(gt and gt[0] != "No ground truth provided for this question." for gt in data["ground_truth"]): # if all ground truths are provided
        metrics_to_evaluate.append(context_recall)
        print("Adding context_recall to metrics as ground truths are provided.")
    else:
        print("Skipping context_recall as not all ground truths are provided or are placeholders.")


    # Ensure RAGAs uses the LLM you intend (e.g. OpenAI by default, or configure a local one)
    # If your chatbot's LLM is local and you want RAGAs to use it,
    # you'll need to wrap it: https://docs.ragas.io/en/latest/how-tos/customisations/custom_llm.html
    # For now, this will default to OpenAI if OPENAI_API_KEY is set.
    
    # RAGAs evaluate expects `answer` and `contexts` columns.
    # If you have ground_truth for metrics like context_recall, ensure it's named `ground_truth` in the dataset.
    # RAGAs 0.1.x and later versions handle asyncio differently.
    # The evaluate function in RAGAs is often async.
    
    try:
        # For RAGAs versions that are async by default
        # For RAGAs 0.1.x you might need `loop = asyncio.get_event_loop()` and `loop.run_until_complete(evaluate(...))`
        # For RAGAs 0.2.x (and similar async versions), evaluate itself might be an awaitable
        result = await asyncio.to_thread(evaluate, dataset=dataset, metrics=metrics_to_evaluate)

        # If evaluate is not async in your RAGAs version or you run into issues:
        # result = evaluate(dataset=dataset, metrics=metrics_to_evaluate)

    except Exception as e:
        print(f"Error during RAGAs evaluation: {e}")
        print("This might be due to asyncio issues with RAGAs version or LLM configuration for RAGAs.")
        print("Trying synchronous evaluation if applicable...")
        try:
            # Fallback for non-async or if the above failed
            result = evaluate(dataset=dataset, metrics=metrics_to_evaluate)
        except Exception as e_sync:
            print(f"Synchronous RAGAs evaluation also failed: {e_sync}")
            return
            
    print("\nRAGAs Evaluation Results:")
    df_results = result.to_pandas()
    print(df_results)

    # Save results to CSV
    results_csv_path = "ragas_evaluation_results.csv"
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nEvaluation results saved to {results_csv_path}")

if __name__ == "__main__":
    # Ensure you have an event loop for asyncio if evaluate is async
    # For RAGAs > 0.1.0, `evaluate` is often async
    # Running the async function
    # Python 3.7+
    asyncio.run(run_evaluation())

    # If you encounter issues with asyncio.run, especially in certain environments like Jupyter:
    # try:
    #     loop = asyncio.get_event_loop()
    #     if loop.is_running():
    #         # This is a common pattern if an event loop is already running (e.g., in Jupyter)
    #         # but can be tricky. `nest_asyncio` can also help here if installed and applied.
    #         print("Async event loop already running. Creating a new task.")
    #         tsk = loop.create_task(run_evaluation())
    #         # loop.run_until_complete(tsk) # This might block if called from within the loop
    #     else:
    #         loop.run_until_complete(run_evaluation())
    # except RuntimeError as e:
    #     if "cannot be called when another loop is running" in str(e):
    #          print("Trying to schedule on existing loop due to RuntimeError")
    #          # This part is tricky and environment-dependent.
    #          # Consider `nest_asyncio.apply()` at the top of your script if you have nested loops.
    #          # For simplicity, if `asyncio.run` fails, you might need to debug the async setup
    #          # or RAGAs specific async handling.
    #          pass # Or re-raise
    #     else:
    #          raise e

</rewritten_file> 