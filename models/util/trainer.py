import torch.nn as nn
from sklearn.metrics import accuracy_score
import torch


def _print_some_examples(model, valid_loader, seq_len, tgt_i2w):
    x_dev_batch, y_dev_batch = iter(valid_loader).next()
    x_dev_batch = x_dev_batch[0:seq_len]
    y_dev_batch = y_dev_batch[0:seq_len]

    out_valid = model.forward(x_dev_batch, y_dev_batch)
    out_valid = torch.argmax(out_valid, dim=2)

    for i in range(seq_len):
        for j in range(len(out_valid[i])):
            token = str(out_valid[i][j].item())

            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'], end='')
                break
            else:
                print(tgt_i2w[token] + " ", end='')
        print()


def train(model, epochs, batch_size, lr, n_class, train_loader, valid_loader, tgt_i2w):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_data = len(train_loader.dataset.X)
    n_dev_data = len(valid_loader.dataset.X)

    # training
    for epoch in range(epochs):
        print("Epochs: {}/{}\n".format(epoch + 1, epochs))
        average_loss = 0
        batch_counter = 1

        for x_batch, y_batch in train_loader:
            print("Batch: {}/{}".format(batch_counter, int(n_data / batch_size)))

            optimizer.zero_grad()

            output = model.forward(x_batch, y_batch)

            loss = criterion(output.view(-1, n_class), y_batch.contiguous().view(-1))
            average_loss += loss

            print("Loss: {}".format(loss))

            loss.backward()
            optimizer.step()

            batch_counter = batch_counter + 1

        # deving
        _print_some_examples(model, valid_loader, 5, tgt_i2w)
        dev_accuracy = 0

        for x_dev_batch, y_dev_batch in valid_loader:
            out_valid = model.forward(x_dev_batch, y_dev_batch).argmax(dim=2).view(-1)

            dev_accuracy += accuracy_score(y_dev_batch.contiguous().view(-1), out_valid)

        print("\nValidation Accuracy: {}".format(dev_accuracy / (n_dev_data / batch_size)))
        print("Average loss: {}\n".format(average_loss / (n_data / batch_size)))