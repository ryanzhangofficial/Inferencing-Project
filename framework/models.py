from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def createModelList():
    m_list = {}
    return m_list

def add(model_list):
    model_name = input("Input a model from Hugging Face: ")

    try:
        print(f"Loading model '{model_name}'...")
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

        model_list[model_name] = {
            "tokenizer": model_tokenizer,
            "model": model
        }

        print(f"Model '{model_name}' added successfully.")
    
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")