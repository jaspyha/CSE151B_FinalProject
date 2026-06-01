# CSE151B_FinalProject
GPU: Nvidia RTX 4070 Laptop GPU (8G VRAM)
Approximate Inference Time: 30 Hours

The code uses VLLM in a virtual environment created using uv. 
Create the uv environment and install the requirements in "requirements.txt". 
The original Hugging Face api key was removed. 
Putting your own may be required to run the code.

The code was originally run in a Jupyter notebook. 
The original notebook is provided as "inference_final.pynb".
"run_inference.py" contains the exported Jupyter notebook.

The code searches in "data" folder for the questions in "public.jsonl" and "private.jsonl"
The code uses "public jsonl" and prints accuracy when SAVE_EVAL=True, else will use "private.jsonl" with no statistics.
The code outputs "vllm_inference_predictions.jsonl" in a created "results" folder then converts to "lastsubpredictions.csv" in the main directory.
