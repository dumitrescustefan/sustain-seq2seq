from models.util.metrics.accuracy import accuracy_score
from models.util.metrics.bleu import bleu_score
from models.util.metrics.rouge import rouge_l_score
from models.util.metrics.meteor import meteor_score
from models.util.metrics.sar import sequence_accuracy_rate
from models.util.utils import clean_sequences
import torch


def evaluate(y_true, y_pred, lookup, cut_at_eos=True, use_accuracy=True, use_bleu=True, use_meteor=True, use_rouge=True, use_sequence_accuracy_rate=True):
    #print("\nEvaluation results:\n")
    if cut_at_eos:
        y_true = clean_sequences(y_true, lookup)
        y_pred = clean_sequences(y_pred, lookup)
        
    eval = {}
    eval["score"] = 0.
    count = 0.
    if use_accuracy:
        accuracy = accuracy_score(y_true, y_pred)
        #print("Accuracy score: {0:.4f}".format(float(accuracy)))
        eval["accuracy"] = accuracy
        eval["score"] += accuracy
        count+=1.
    if use_bleu:
        bleu = bleu_score(y_true, y_pred, lookup)
        #print("Bleu score: {0:.4f}".format(bleu))
        eval["bleu"] = bleu
        eval["score"] += bleu
        count+=1.
    if use_meteor:
        meteor = meteor_score(y_true, y_pred, lookup)
        #print("Meteor score: {0:.4f}".format(meteor))
        eval["meteor"] = meteor
        eval["score"] += meteor
        count+=1.
    if use_rouge:
        rouge_r, rouge_p, rouge_f = rouge_l_score(y_true, y_pred, lookup)
        #print("Rouge-l score: recall-{0:.4f} precision-{0:.4f} f1-{0:.4f}".format(rouge_r, rouge_p, rouge_f))
        eval["rouge_l_r"] = rouge_r
        eval["rouge_l_p"] = rouge_p
        eval["rouge_l_f"] = rouge_f
        eval["score"] += rouge_f
        count+=1.
    if use_sequence_accuracy_rate:
        eval["sar"] = sequence_accuracy_rate(y_true, y_pred)
    
    eval["score"] = eval["score"]/count
    return eval["score"], eval

"""
if __name__ == "__main__":
    y_true = torch.Tensor([[2, 3, 5, 5], [5, 3, 1, 2]])
    y_pred = torch.Tensor([[2, 3, 5, 5], [5, 2, 1, 2]])
    #i2w = {'2': "uuu", '3': "da", '5': "baba", '1':"cc"} it is a list!!

    print(accuracy_score(y_true, y_pred))
    print(bleu_score(y_true, y_pred, i2w))
    print(meteor_score(y_true, y_pred, i2w))

    print(rouge_l_score(y_true, y_pred, i2w))
"""