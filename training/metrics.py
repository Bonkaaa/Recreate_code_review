from evaluate import load
from Bleu_score.bleu import *
from Bleu_score.tokenizer_13a import Tokenizer13a

def calculate_bleu_score(actuals, preds):
    """
    Calculating bleu_socre
    :param actuals: references
    :param preds: candidates

    :return: BLEU score
    """

    #Weights

    bleu_score = compute_bleu(translation_corpus = preds, reference_corpus = actuals ,smooth = True)

    return bleu_score

def calculate_exact_match_score(actuals, preds):
    """
    Calculating bleu_socre
    :param actuals: references
    :param preds: candidates

    :return: EM score
    """
    exact_match_metric = load("exact_match")
    exact_match_metric.add_batch(predictions=preds, references=actuals)
    exact_match_score = exact_match_metric.compute(ignore_case = True)

    return exact_match_score['exact_match']

if __name__ == '__main__':
    # Correctly tokenized test case
    ref = [
        "hello there general kenobi",
        "foo bar foobar"]
    candidate = [
        "hello there general kenobi",
        "foo bar foobar"]

    # Compute BLEU score
    score = calculate_bleu_score(ref, candidate)
    print(f"BLEU Score: {score}")

    # Verify correctness
    bleu_score = compute_bleu(translation_corpus=candidate, reference_corpus=ref, smooth = True)
    print(f"Expected BLEU Score: {bleu_score}")

    # # Compute EM score
    # EM_score = calculate_exact_match_score(ref, candidate)['exact_match']
    # print(f"EM Score: {EM_score:.4f}")