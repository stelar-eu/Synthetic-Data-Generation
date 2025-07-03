

# # 🧠 Ollama Summary Generator with Shutdown
# This notebook lets you:
# - Choose a model and prompt style
# - Generate summaries by extrating text from a .CSV file
# - Optionally shut down the Ollama server from within Jupyter Notebook
# 
# **Note:** Make sure `ollama serve` is running in your Terminal before running this notebook.


# Step 1: Import required libraries
import requests
import time
from datetime import datetime
import subprocess
from IPython.display import display, clear_output
from ipywidgets import Dropdown, Button, Output, VBox
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
        recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay)


# Step 2: Configure Models and Prompts
models = ['mistral', 'llama2', 'tinyllama'] #Add more models as required
prompts = ['one-shot', 'few-shot', 'cot'] ##Add more prompts as required

model_dropdown = Dropdown(options=models, value='mistral', description='Model:')
prompt_dropdown = Dropdown(options=prompts, value='cot', description='Prompt:')

display(VBox([model_dropdown, prompt_dropdown]))


# Step 3: Load input CSV with 'description' column
input_csv = "input.csv"
df= pd.read_csv(input_csv)#.head(4)
assert 'columns_name' in df.columns, "❌ 'columns_name' column not found in the CSV."
df.head()


# ### Task- Summarization 


# Step 4: Define Prompt Templates (tweak as necessary)
prompt_templates = {
    "one-shot": "Summarize the following report clearly and concisely:\n\n{text}\n\nSummary:",
    "few-shot": (
        "Here is how you summarize reports:\n"
        "Example:\nInput: A fire broke out in the kitchen due to a gas leak.\n"
        "Summary: Kitchen fire caused by gas leak.\n\nNow summarize:\n{text}\n\nSummary:"
    ),
    "cot": (
        "Read the report carefully and write a well-structured, fluent summary using natural language paragraphs only. "
        "Do not use any numbering, bullet points, or formatting. Maintain the original meaning, but phrase it concisely "
        "in your own words without any extra commentary.\n\n"
        "Report:\n{text}\n\nSummary:"
    ),
}


#### Function for Generating Summaries


# Step 5: Define the summarization function (with streaming fix and fallback)
def run_summary(model, prompt_type, filename):
    import json

    prompt_template = prompt_templates[prompt_type]
    summaries = []
    times = []
    start_all = time.time()

    for i, text in enumerate(df['columns_name']):
        summary = "ERROR"  # fallback assignment
        try:
            start = time.time()
            prompt = prompt_template.format(text=text)

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt},
                stream=True
            )
            response.raise_for_status()

            # Collect streamed JSON output
            parts = []
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            parts.append(data["response"])
                    except Exception as parse_err:
                        print(f"⚠️ Stream parse error at row {i+1}: {parse_err}")

            summary = "".join(parts).strip()

        except Exception as e:
            print(f"⚠️ Error at row {i+1}: {e}")

        summaries.append(summary)
        times.append(round(time.time() - start, 2))
        print(f"{i+1}/{len(df)} complete.")

    df[f"{model}_{prompt_type}_summary"] = summaries
    df[f"{model}_{prompt_type}_time"] = times

    output_file = f"summaries_{model}_{prompt_type}_{input_csv}"
    df.to_csv(output_file, index=False)
    total_time = round(time.time() - start_all, 2)
    print(f"✅ Saved to {output_file}")
    print(f"⏱️ Total time: {total_time}s")


# Step 6: Run the summary generator with selected model and prompt
run_summary(model_dropdown.value, prompt_dropdown.value,input_csv)


# Step 7: Optional — Shut down the Ollama server
def stop_ollama():
    try:
        subprocess.run("kill $(lsof -ti :11434)", shell=True, check=True)
        print("🛑 Ollama server stopped.")
    except Exception as e:
        print(f"⚠️ Could not stop Ollama: {e}")

# Uncomment to stop Ollama from Jupyter
stop_ollama()

