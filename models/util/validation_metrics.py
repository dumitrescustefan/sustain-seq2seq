from models.util.metrics.accuracy import accuracy_score
from models.util.metrics.bleu import bleu_score
from models.util.metrics.rouge import rouge_l_score
from models.util.metrics.meteor import meteor_score

import torch


def evaluate(y_true, y_pred, i2w, show_accurracy=True, show_bleu=True, show_meteor=True, show_rogue=True):
    print("\nEvaluation results:\n")

    if show_accurracy:
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy score: {0:.4f}".format(float(accuracy)))
    if show_bleu:
        bleu = bleu_score(y_true, y_pred, i2w)
        print("Bleu score: {0:.4f}".format(bleu))
    if show_meteor:
        meteor = meteor_score(y_true, y_pred, i2w)
        print("Meteor score: {0:.4f}".format(meteor))
    if show_rogue:
        rouge_r, rouge_p, rouge_f = rouge_l_score(y_true, y_pred, i2w)
        print("Rogue-l score: recall-{0:.4f} precision-{0:.4f} f1-{0:.4f}".format(rouge_r, rouge_p, rouge_f))


if __name__ == "__main__":
    y_true = torch.Tensor([[2, 3, 5, 5], [5, 3, 1, 2]])
    y_pred = torch.Tensor([[2, 3, 5, 5], [5, 3, 2, 2]])
    i2w = {'2': "uuu", '3': "da", '5': "baba", '1': "didi"}

    print(bleu_score(y_true, y_pred, i2w))
    print(meteor_score(y_true, y_pred, i2w))

    print(rouge_l_score(y_true, y_pred, i2w))
