import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, n_class, n_emb_dim, n_hidden, n_lstm_units, n_lstm_dropout, device):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(n_class, n_emb_dim)
        self.lstm = nn.LSTM(n_emb_dim, n_hidden, n_lstm_units, dropout=n_lstm_dropout,
                            bidirectional=True, batch_first=True)

        self.to(device)

    def forward(self, input):
        embeddings = self.embedding_layer(input)

        output, states = self.lstm(embeddings)

        return output, states


class AttnDecoder(nn.Module):
    def __init__(self, n_class, n_emb_dim, n_hidden, n_lstm_units, n_lstm_dropout, n_dropout, device):
        super(AttnDecoder, self).__init__()
        self.embedding_layer = nn.Embedding(n_class, n_emb_dim)
        self.attn1 = nn.Linear(n_hidden*n_lstm_units * 2 + n_hidden * 2, n_hidden * 4)
        self.attn2 = nn.Linear(n_hidden * 4, 1)
        self.dropout = nn.Dropout(n_dropout)
        self.lstm = nn.LSTM(n_emb_dim + n_hidden * 2, n_hidden, n_lstm_units, dropout=n_lstm_dropout,
                            bidirectional=True, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        self.output_layer = nn.Linear(n_hidden * 2, n_class)

        self.to(device)

    def _calculate_context_vector(self, state_h, enc_output):
        state_h = state_h.permute(1, 0, 2).reshape(enc_output.shape[0], 1, -1).expand(-1, enc_output.shape[1], -1)

        attn_input = torch.cat((enc_output, state_h), dim=2)
        attn_hidden = self.attn1(attn_input)
        attn_hidden = self.dropout(attn_hidden)
        attn_output = self.attn2(attn_hidden)
        attn_weights = self.softmax(attn_output)

        context_vector = torch.mul(enc_output, attn_weights)
        return torch.sum(context_vector, 1)

    def forward(self, input, enc_output, enc_states):
        embeddings = self.embedding_layer(input)

        context_vector = self._calculate_context_vector(enc_states[0], enc_output)
        lstm_input = torch.cat((embeddings[:, 0, :], context_vector), dim=1).reshape(enc_output.shape[0], 1, -1)
        dec_output, dec_states = self.lstm(lstm_input, enc_states)

        for i in range(1, input.shape[1]):
            context_vector = self._calculate_context_vector(dec_states[0], enc_output)
            lstm_input = torch.cat((embeddings[:, i, :], context_vector), dim=1).reshape(enc_output.shape[0], 1, -1)
            curr_dec_output, dec_states = self.lstm(lstm_input, dec_states)
            dec_output = torch.cat((dec_output, curr_dec_output), dim=1)

        output = self.output_layer(dec_output)

        return output


class LSTMAttnEncoderDecoder(nn.Module):
    def __init__(self, n_class, n_emb_dim, n_hidden, n_lstm_units, n_lstm_dropout, n_dropout):
        super(LSTMAttnEncoderDecoder, self).__init__()

        if torch.cuda.is_available():
            print('Running on GPU.')
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            print('Running on CPU.')
            self.cuda = False
            self.device = torch.device('cpu')

        self.encoder = Encoder(n_class, n_emb_dim, n_hidden, n_lstm_units, n_lstm_dropout, self.device)
        self.decoder = AttnDecoder(n_class, n_emb_dim, n_hidden, n_lstm_units, n_lstm_dropout, n_dropout, self.device)

        if self.cuda:
            self.to(self.device)

    def forward(self, x, y):
        enc_output, enc_states = self.encoder.forward(x)
        # print(enc_states)
        output = self.decoder.forward(y, enc_output, enc_states)

        return output
