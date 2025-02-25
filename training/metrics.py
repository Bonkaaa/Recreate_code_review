from evaluate import load

def calculate_bleu_score(actuals, preds):
    """
    Calculating bleu_socre
    :param actuals: references
    :param preds: candidates

    :return: BLEU score
    """

    #Weights

    bleu = load("bleu")
    bleu.add(predictions=preds, references=actuals)
    bleu_score = bleu.compute(predictions=preds, references=actuals, smooth = True)

    return bleu_score['bleu']

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
    print(f"BLEU Score: {score:.4f}")

    bleu = load("bleu")
    # Verify correctness
    bleu_score = bleu.compute(predictions=candidate, references=ref, smooth = True)
    print(f"Expected BLEU Score: {bleu_score['bleu']:.4f}")

    # # Compute EM score
    # EM_score = calculate_exact_match_score(ref, candidate)['exact_match']
    # print(f"EM Score: {EM_score:.4f}")