from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import evaluate
from evaluate import load

def calculate_bleu_score(actuals, preds, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculating bleu_socre
    :param actuals: references
    :param preds: candidates
    :param weights: weights for n-grams

    :return: BLEU score
    """

    smoothing = SmoothingFunction().method1  # Avoid zeros score for missing n-grams

    bleu_score = corpus_bleu(actuals, preds, weights=weights, smoothing_function=smoothing)

    return bleu_score

def calculate_exact_match_score(actuals, preds):
    """
    Calculating bleu_socre
    :param actuals: references
    :param preds: candidates

    :return: EM score
    """
    exact_match_metric = load("exact_match")
    exact_match_metric.add_batch(predictions= preds, references= actuals)
    exact_match_score = exact_match_metric.compute(predictions = preds, references = actuals, ignore_case = True)

    return exact_match_score

if __name__ == '__main__':
    # Correctly tokenized test case
    ref = [["the", "picture", "is", "clicked", "by", "me"],
           ["the", "picture", "was", "clicked", "by", "me"]]        # List of lists (each reference should be tokenized)
    candidate = [["the", "picture", "the", "picture", "by", "me"],
                 ["the", "picture", "a", "picture", "in", "me"]]    # Tokenized candidate sentence

    # Compute BLEU score
    score = calculate_bleu_score(ref, candidate)
    print(f"BLEU Score: {score:.4f}")

    # Verify correctness
    expected_score = corpus_bleu(ref, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    print(f"Expected BLEU Score: {expected_score:.4f}")

    # # Compute EM score
    # EM_score = calculate_exact_match_score(ref, candidate)['exact_match']
    # print(f"EM Score: {EM_score:.4f}")