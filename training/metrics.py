from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_metrics(actuals, preds, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculating bleu_socre
    :param actuals: the output of the model
    :param preds: the target that we want to output
    :param weights: weights for n-grams

    :return: BLEU score
    """

    smoothing = SmoothingFunction().method1  # Avoid zeros score for missing n-grams

    bleu_score = sentence_bleu(actuals, preds, weights=weights, smoothing_function=smoothing)

    return bleu_score

if __name__ == '__main__':
    # Correctly tokenized test case
    ref = [["the", "picture", "is", "clicked", "by", "me"],["the", "picture", "was", "clicked", "by", "me"]] # List of lists (each reference should be tokenized)
    candidate = ["the", "picture", "the", "picture", "by", "me"]  # Tokenized candidate sentence

    # Compute BLEU score
    score = calculate_metrics(ref, candidate)
    print(f"BLEU Score: {score:.4f}")

    # Verify correctness
    expected_score = sentence_bleu(ref, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    print(f"Expected BLEU Score: {expected_score:.4f}")