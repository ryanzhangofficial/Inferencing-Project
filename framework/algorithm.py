import math
import random
import sys
import os
from utilities.models import inference    
from utilities.metrics import calculate_bleu
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import fastText

# Algorithm for model selection framework
#
# @input T = total amount of requests
# @input c = constant hyperparameter c > 1
# p_t = probability of querying all models/exploration
# X_t = binary value, explore or not explore
# t_i = input sample for current request
# t_c = content for current request derived from t_i
# l, s = large model, small model <!--In a real world setting, this should be scalable-->
def algorithm(T, c, m_list):
    l_predictor, s_predictor = None, None
    for t in range(1, T+1):
        t_i = input(f"Enter input sample for request {t}/{T}: ")
        print(t_i)

        p_t = min(1, c/math.sqrt(t))
        X_t = Bernoulli(p_t)
        if X_t == 1:
            t_e = input("Input expected outcome: ") # wmt14_dataset = load_dataset('wmt14', 'de-en', split='validation')
            print("Received Input")
            t_c = getResults(t_i, t_e, m_list)
            print("Got Results")
            l_predictor, s_predictor = sgdStep(t_c)
            l_predictor, s_predictor = checkpoint(l_predictor, s_predictor)
            print("SGD STEP!")
        else:
            output = queryBest(t_i, m_list, l_predictor, s_predictor)
            print(output)

def Bernoulli(p_t):
    return random.random() < p_t 

def queryBest(t_i, m_list, l_predictor, s_predictor):
    l_acc, s_acc = predict(t_i, l_predictor, s_predictor)
    if l_acc > s_acc: # use larger model, e.g. 7b llama
        return inference(t_i, m_list['meta-llama/Llama-2-13b-chat-hf']['model'], m_list['meta-llama/Llama-2-13b-chat-hf']['tokenizer'])
    else: # use smaller model for all other cases, e.g. 3b llama
        return inference(t_i, m_list['meta-llama/Llama-2-7b-chat-hf']['model'], m_list['meta-llama/Llama-2-7b-chat-hf']['tokenizer'])

def sgdStep(t_c):
    with open("fasttext_large.txt", "w") as f:
        f.write(f"__label__{t_c['large_model_accuracy']} {t_c['input_text']}\n")
    with open("fasttext_small.txt", "w") as f:
        f.write(f"__label__{t_c['small_model_accuracy']} {t_c['input_text']}\n")

    if os.path.exists("large_predictor.bin"):
        large_model_predictor = fastText.train_supervised(input="fasttext_large.txt", epoch=1, lr=1.0, wordNgrams=2, inputModel="large_predictor.bin")
    else:
        large_model_predictor = fastText.train_supervised(input="fasttext_large.txt", epoch=1, lr=1.0, wordNgrams=2)

    if os.path.exists("small_predictor.bin"):
        small_model_predictor = fastText.train_supervised(input="fasttext_small.txt", epoch=1, lr=1.0, wordNgrams=2, inputModel="small_predictor.bin")
    else:
        small_model_predictor = fastText.train_supervised(input="fasttext_small.txt", epoch=1, lr=1.0, wordNgrams=2)

    return large_model_predictor, small_model_predictor

def predict(text, l_predictor, s_predictor):
    l_predicted_label = l_predictor.predict(text)[0][0]
    l_predicted_accuracy = int(l_predicted_label.replace('__label__', ''))

    s_predicted_label = s_predictor.predict(text)[0][0]
    s_predicted_accuracy = int(s_predicted_label.replace('__label__', ''))

    return l_predicted_accuracy, s_predicted_accuracy

def getResults(t_i, t_e, m_list):
    s_output = inference(t_i, m_list['microsoft/Phi-3-mini-4k-instruct']['model'], m_list['microsoft/Phi-3-mini-4k-instruct']['tokenizer'])
    print("LARGE INFERENCE DONE")
    l_output = inference(t_i, m_list['microsoft/Phi-3-small-8k-instruct']['model'], m_list['microsoft/Phi-3-small-8k-instruct']['tokenizer'])    
    print("SMALL INFERENCE DONE")

    s_acc = calculate_bleu(s_output, t_e)
    l_acc = calculate_bleu(l_output, t_e)

    return {
        'input_text': t_i,
        'large_model_accuracy': l_acc,
        'small_model_accuracy': s_acc
    }

def checkpoint(l_predictor, s_predictor, t, p_t):
    if p_t > 0.5:
        if t % int(math.sqrt(t)) == 0:
            l_predictor, s_predictor = save_models(l_predictor, s_predictor)
    else:
        if t % max(1, int(1 / math.sqrt(t))) == 0:
            l_predictor, s_predictor = save_models(l_predictor, s_predictor)

    return l_predictor, s_predictor

def save_models(l_predictor, s_predictor):
    if l_predictor:
        l_predictor.save_model("large_predictor.bin")
    if s_predictor:
        s_predictor.save_model("small_predictor.bin")
    print("Models saved successfully.")

    return l_predictor, s_predictor