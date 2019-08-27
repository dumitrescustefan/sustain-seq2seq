from sklearn.metrics import accuracy_score as sk_accuracy_score


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    y_true_flat, y_pred_flat = ([index for y_true_seq in y_true for index in y_true_seq],
                                [index for y_pred_seq in y_pred for index in y_pred_seq])

    return sk_accuracy_score(y_true_flat, y_pred_flat, normalize, sample_weight)