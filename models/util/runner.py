import torch


def run(model, test_loader, src_i2w, tgt_i2w):
    y = torch.Tensor([[2]]).long()
    model.eval()
    max_pred_len = 150

    for X, _ in test_loader:
        for i in range(X.shape[1]):
            token = str(X[0][i].item())

            if token == '3':
                break

            print(src_i2w[token], end='')

        print()

        for _ in range(max_pred_len):
            y_pred = model.forward(X, y)[:, -1:, :]
            token = str(torch.argmax(y_pred, dim=2).item())
            print(tgt_i2w[token], end='')

            if token == '3':
                break

            y_pred = torch.Tensor([[int(token)]]).long()
            y = torch.cat((y, y_pred), dim=1)

        print("-" * 200)
        print("\n")
