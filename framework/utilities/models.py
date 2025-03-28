from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def createModelList():
    m_list = {}
    return m_list

def createDefaultModelList():
    m_list = {}

    model_name_1 = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1, trust_remote_code=True)
    model_1 = AutoModelForCausalLM.from_pretrained(model_name_1, 
                                                   torch_dtype=torch.float16, 
                                                   trust_remote_code=True, 
                                                   attn_implementation='eager').to("cpu") #.to("cuda")
    m_list[model_name_1] = {
        "tokenizer": tokenizer_1,
        "model": model_1
    }

    model_name_2 = "microsoft/Phi-3-small-8k-instruct"
    tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2, trust_remote_code=True)
    model_2 = AutoModelForCausalLM.from_pretrained(model_name_2, 
                                                   torch_dtype=torch.float16, 
                                                   trust_remote_code=True, 
                                                   attn_implementation='eager').to("cpu") #.to("cuda")
    m_list[model_name_2] = {
        "tokenizer": tokenizer_2,
        "model": model_2
    }

    return m_list

def inference(t_i, model, tokenizer):
    inputs = tokenizer(t_i, return_tensors="pt") # .to("cuda")
    print("DONE TOKENIZING!")
    output = model.generate(inputs['input_ids'], max_length=50)
    print("DONE GENERATING!")
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return output_text

# for scalability latter
def add(model_list):
    model_name = input("Input a model from Hugging Face: ")

    try:
        print(f"Loading model '{model_name}'...")
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)#.to("cuda")

        model_list[model_name] = {
            "tokenizer": model_tokenizer,
            "model": model
        }

        print(f"Model '{model_name}' added successfully.")
    
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")