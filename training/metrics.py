from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_metrics(actuals, preds, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculating bleu_socre
    :param actuals: the output of the model
    :param preds: the target that we want to output
    :param weights: weights for n-grams

    :return: BLEU score
    """
    predicts_list = [preds]  # Convert to list
    smoothing = SmoothingFunction().method1  # Avoid zeros score for missing n-grams

    bleu_score = sentence_bleu(predicts_list, actuals, weights=weights, smoothing_function=smoothing)

    return bleu_score