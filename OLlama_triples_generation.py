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

# Step 2: Configure Models and Prompts
models = ['mistral', 'llama2', 'tinyllama']  # Add more models as required
prompts = ['one-shot', 'few-shot', 'cot']  # Add more prompts as required

model_dropdown = Dropdown(options=models, value='mistral', description='Model:')
prompt_dropdown = Dropdown(options=prompts, value='cot', description='Prompt:')

display(VBox([model_dropdown, prompt_dropdown]))

# Step 3: Load input CSV with 'hazard_category' and 'product_category' columns
input_csv = "input.csv"
df = pd.read_csv(input_csv)
assert 'hazard_category' in df.columns, "❌ 'hazard_category' column not found in the CSV."
assert 'product_category' in df.columns, "❌ 'product_category' column not found in the CSV."
df.head()

# Step 4: Define Prompt Templates for Triple Generation
prompt_templates = {
    "one-shot": (
        "Generate 20 semantic triples in the form (subject, relation, object) based on the following category:\n\n"
        "Category: {category}\n\n"
        "Each triple should reflect real-world knowledge associated with the category. Use lowercase for general terms, "
        "and preserve capitalization for proper nouns or acronyms (e.g., FDA, USDA).\n\n"
        "Output the triples as tab-separated values, one per line."
    ),
    "few-shot": (
        "Here are some examples of semantic triples for the category \"fire\":\n\n"
        "fire\tcaused by\telectrical short-circuit\n"
        "fire\tstarts in\tkitchen\n"
        "fire\textinguished by\twater\n"
        "fire\tresults in\tproperty damage\n"
        "fire\tspreads through\tflammable materials\n\n"
        "Now generate 20 semantic triples for the category below in the same format (subject, relation, object), using "
        "tab-separated values, one per line.\n\n"
        "Category: {category}"
    ),
    "cot": (
        "Think step by step about common scenarios, causes, consequences, and properties associated with the following category:\n\n"
        "Category: {category}\n\n"
        "First, reflect briefly on real-world events or knowledge associated with this category. Then, generate 20 semantic triples "
        "(subject, relation, object) representing key facts or relations. Each triple should be tab-separated and written in lowercase "
        "unless it's a proper noun or acronym.\n\n"
        "Do not include any explanations or reasoning in the output—just the 20 tab-separated triples."
    ),
}

# Step 5: Define the Triple Generation Function
def generate_triples(model, prompt_type, filename):
    prompt_template = prompt_templates[prompt_type]
    categories = {
        "hazard": df["hazard_category"].dropna().unique(),
        "product": df["product_category"].dropna().unique()
    }

    records = []
    start_all = time.time()

    for category_type, values in categories.items():
        for i, category in enumerate(values):
            triples_text = "ERROR"
            try:
                prompt = prompt_template.format(category=category)

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt},
                    stream=True
                )
                response.raise_for_status()

                # Stream LLM response
                parts = []
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "response" in data:
                                parts.append(data["response"])
                        except Exception as parse_err:
                            print(f"⚠️ Stream parse error at {category_type} '{category}': {parse_err}")

                triples_text = "".join(parts).strip()

                # Parse triples into (subject, relation, object)
                for line in triples_text.splitlines():
                    if line.count('\t') == 2:
                        subject, relation, obj = line.strip().split('\t')
                        records.append({
                            "category_type": category_type,
                            "category_name": category,
                            "subject": subject.strip(),
                            "relation": relation.strip(),
                            "object": obj.strip()
                        })

            except Exception as e:
                print(f"⚠️ Error processing {category_type} '{category}': {e}")

            print(f"{category_type} category {i+1}/{len(values)} complete.")

    # Save output
    triples_df = pd.DataFrame(records)
    output_file = f"triples_{model}_{prompt_type}_{filename}"
    triples_df.to_csv(output_file, index=False, sep="\t")
    print(f"✅ Saved triples to {output_file}")
    print(f"⏱️ Total time: {round(time.time() - start_all, 2)}s")

# Step 6: Run the triple generator with selected model and prompt
generate_triples(model_dropdown.value, prompt_dropdown.value, input_csv)

# Step 7: Optional — Shut down the Ollama server
def stop_ollama():
    try:
        subprocess.run("kill $(lsof -ti :11434)", shell=True, check=True)
        print("🛑 Ollama server stopped.")
    except Exception as e:
        print(f"⚠️ Could not stop Ollama: {e}")

# Uncomment to stop Ollama from Jupyter
# stop_ollama()
