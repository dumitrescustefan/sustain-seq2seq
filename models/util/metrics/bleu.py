from nltk.translate.bleu_score import corpus_bleu


def bleu_score(y_true, y_pred, lookup, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False):
    references = list()
    hypothesis = list()

    for y_true_seq, y_pred_seq in zip(y_true, y_pred):
        # Adds a new translation of a sentence to the references/hypothesis. Uses the vocabulary to map the indexes. References: [n_samples, 1, seq_len]. Hypothesis: [n_samples, seq_len].
        #references.append([[i2w[str(int(index))] for index in y_true_seq]])
        #hypothesis.append([i2w[str(int(index))] for index in y_pred_seq])
        references.append([[lookup.convert_ids_to_tokens(index) for index in y_true_seq]])
        hypothesis.append([lookup.convert_ids_to_tokens(index) for index in y_pred_seq])

    return corpus_bleu(references, hypothesis, weights, smoothing_function, auto_reweigh)