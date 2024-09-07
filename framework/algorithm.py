import math
import random
import fastText 
import time
import os

# Algorithm for model selection framework
#
# @input T = total amount of requests
# @input c = constant hyperparameter c > 1
# p_t = probability of querying all models/exploration
# X_t = binary value, explore or not explore
# t_i = input sample for current request
# t_c = content for current request derived from t_i
# l, s = large model, small model <!--In a real world setting, this should be scalable-->
def algorithm(T, c):
    l_predictor, s_predictor = None, None
    for t in range(1, T+1):
        t_i = input(f"Enter input sample for request {t}/{T}: ")

        p_t = min(1, c/math.sqrt(t))
        X_t = Bernoulli(p_t)
        if X_t == 1:
            t_c = getResults(t_i)
            sgdStep(t_c)
        else:
            l, s = queryBest(t_i, l_predictor, s_predictor)

def Bernoulli(p_t):
    return random.random() < p_t 

def queryBest(t_i):
    l_acc, s_acc = predict(t_i)
    if l_acc > s_acc: # use larger model, e.g. 60b llama
        return "" 
    else: # use smaller model for all other cases, e.g. 7b llama
        return ""

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

def getResults(t_i):
    # input t_i(the input sample for the request) to 

    return {
        'input_text': t_i,
        'large_model_accuracy': l_acc,
        'small_model_accuracy': s_acc
    }