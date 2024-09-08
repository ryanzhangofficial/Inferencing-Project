import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_bleu(generated_output, reference_output):
    generated_tokens = nltk.word_tokenize(generated_output)
    reference_tokens = [nltk.word_tokenize(reference_output)]
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smoothie)
    return bleu_score * 100

def calculate_rouge(generated_output, reference_output):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_output, generated_output)
    return {
        'rouge1': scores['rouge1'].fmeasure * 100,
        'rouge2': scores['rouge2'].fmeasure * 100,
        'rougeL': scores['rougeL'].fmeasure * 100
    }

def calculate_exact_match(generated_output, reference_output):
    if generated_output.strip() == reference_output.strip():
        return 100.0
    else:
        return 0.0