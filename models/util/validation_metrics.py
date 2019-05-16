from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu
import torch


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    return accuracy_score(y_true, y_pred, normalize, sample_weight)


def bleu_score(y_true, y_pred, i2w, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False):
    references = list()
    hypothesis = list()

    for y_true_seq, y_pred_seq in zip(y_true, y_pred):
        # Adds a new translation of a sentence to the references/hypothesis. Uses the vocabulary to map the indexes. References: [n_samples, 1, seq_len]. Hypothesis: [n_samples, seq_len].
        references.append([[i2w[index.item()] for index in y_true_seq]])
        hypothesis.append([i2w[index.item()] for index in y_pred_seq])

    return corpus_bleu(references, hypothesis, weights, smoothing_function, auto_reweigh)



if __name__ == "__main__":
    y_true = torch.Tensor([[2, 3, 5, 5], [5, 3, 1, 2]])
    y_pred = torch.Tensor([[2, 3, 5, 5], [5, 3, 2, 2]])
    i2w = {2: "uuu", 3: "da", 5: "baba", 1: "didi"}

    print(bleu_score(y_true, y_pred, i2w))