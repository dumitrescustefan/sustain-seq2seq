
def sequence_error_rate(y_true, y_pred):
    total = len(y_true)*1.
    correct = 0.
    for seq_true, seq_pred in zip(y_true, y_pred):
        # print(seq_true)
        # print(seq_pred)
        # print()
        if len(seq_true) == len(seq_pred):
            match = True
            for i in range(len(seq_true)):
                if seq_true[i] != seq_pred[i]:
                    match = False
                    break
            if match:
                correct += 1.
    
    return correct/total
