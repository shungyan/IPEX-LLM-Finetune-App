import sys
import json
import argparse

from transformers import AutoTokenizer
from template import ollama_template

MODELFILE_TEMPLATE = '''FROM {save_path}

TEMPLATE """{chat_template}"""
'''

def create_modelfile(model_path, save_path):
    with open(f"{model_path}/config.json", 'r') as f:
        _data = f.read()
        model_data = json.loads(_data)

    chat_template = ollama_template[model_data['model_type']]
    data = MODELFILE_TEMPLATE.format(chat_template=chat_template, save_path=save_path)
    with open(save_path, "w") as f:
        f.write(data)

import subprocess

def create_ollama_model(save_path, quant_format, ollama_custom_model_id):
    command = [
        "ollama",
        "create",
        "-f", save_path,
        "--quantize", quant_format,
        ollama_custom_model_id
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Model creation output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error creating model:", e.stderr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a modified model file from a tokenizer.')
    parser.add_argument('--model_path', type=str, required=True, help='The identifier for the model file')
    parser.add_argument('--save_path', type=str, required=True, help='The path where the modified model file will be saved')
    parser.add_argument('--quant_format', type=str, default="Q4_K_M")
    parser.add_argument('--ollama_custom_model_id', type=str, required=True,help='Name of ollama model' )
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    
    create_modelfile(args.model_path, args.save_path)
    create_ollama_model( args.save_path, args.quant_format, args.ollama_custom_model_id)