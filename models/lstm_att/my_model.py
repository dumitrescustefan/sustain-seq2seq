import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, n_classes, n_emb_dim, n_hidden, n_lstm_units):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(n_classes, n_emb_dim)
        self.lstm = nn.LSTM(n_emb_dim, n_hidden, n_lstm_units, dropout=0.2, batch_first=True)

    def forward(self, input):
        embeddings = self.embedding_layer(input)

        output, states = self.lstm(embeddings)

        # state_h = None
        # prev_states = None
        #
        # for i in range(input.shape[1]):
        #     if i is 0:
        #         _, (curr_state_h, curr_state_c) = self.lstm(embeddings[:, i, :].reshape(embeddings.shape[0], 1,
        #                                                                                 embeddings.shape[2]))
        #         prev_states = (curr_state_h, curr_state_c)
        #         curr_state_h = curr_state_h.reshape(1, curr_state_h.shape[0], curr_state_h.shape[1],
        #                                             curr_state_h.shape[2])
        #         state_h = curr_state_h
        #     else:
        #         _, (curr_state_h, curr_state_c) = self.lstm(embeddings[:, i, :].reshape(embeddings.shape[0], 1,
        #                                                                                 embeddings.shape[2]),
        #                                                     prev_states)
        #         prev_states = (curr_state_h, curr_state_c)
        #         curr_state_h = curr_state_h.reshape(1, curr_state_h.shape[0], curr_state_h.shape[1],
        #                                             curr_state_h.shape[2])
        #         state_h = torch.cat((state_h, curr_state_h), dim=0)

        return output, states


class AttnDecoder(nn.Module):
    def __init__(self, n_classes, n_emb_dim, n_hidden, n_lstm_units):
        super(AttnDecoder, self).__init__()
        self.embedding_layer = nn.Embedding(n_classes, n_emb_dim)
        self.attn1 = nn.Linear(n_hidden * 2, n_hidden * 3)
        self.drop1 = nn.Dropout(0.2)
        self.attn2 = nn.Linear(n_hidden * 3, n_hidden * 3)
        self.drop2 = nn.Dropout(0.2)
        self.attn3 = nn.Linear(n_hidden * 3, 1)
        self.lstm = nn.LSTM(n_emb_dim, n_hidden, n_lstm_units, dropout=0.2, batch_first=True)
        self.output_layer = nn.Linear(n_hidden*2, n_classes)

    # def _calculate_context_vector(self, embedding, enc_output, enc_states):
    #     dec_output, dec_states = self.lstm(embedding.reshape(embedding.shape[0], 1, embedding.shape[1]), enc_states)
    #
    #     attn_input = torch.cat((enc_output, dec_output.expand(-1, enc_output.shape[1], -1)), dim=2)
    #     attn_hidd = self.attn1(attn_input)
    #     attn_out = torch.squeeze(self.attn2(attn_hidd), dim=2)
    #
    #     attn_weights = nn.functional.softmax(attn_out, dim=1)
    #     context_vector = torch.mul(enc_output, attn_weights.reshape(attn_weights.shape[0], attn_weights.shape[1], 1))
    #     context_vector = torch.sum(context_vector, dim=1)
    #
    #     return dec_output, \
    #            context_vector.reshape(context_vector.shape[0], 1, context_vector.shape[1]), \
    #            dec_states

    # def forward(self, input, enc_output, enc_states):
    #     embeddings = self.embedding_layer(input)
    #
    #     dec_output, context_vector, dec_states = self._calculate_context_vector(
    #         embeddings[:, 0, :], enc_output, enc_states)
    #
    #     for i in range(1, input.shape[1]):
    #         curr_dec_out, curr_context_vector, dec_states = self._calculate_context_vector(
    #             embeddings[:, i, :], enc_output, dec_states)
    #
    #         context_vector = torch.cat((context_vector, curr_context_vector), dim=1)
    #         dec_output = torch.cat((dec_output, curr_dec_out), dim=1)
    #
    #     ff_input = torch.cat((dec_output, context_vector), dim=2)
    #     output = self.output_layer(ff_input)
    #
    #     return output

    def forward(self, input, enc_output, enc_states):
        embeddings = self.embedding_layer(input)

        dec_output, _ = self.lstm(embeddings, enc_states)

        context_vectors = torch.Tensor(dec_output.shape)

        for i in range(dec_output.shape[1]):
            curr_dec_output = dec_output[:, i, :].reshape(dec_output.shape[0], 1, dec_output.shape[2]).expand(-1, enc_output.shape[1], -1)

            attn_input = torch.cat((enc_output, curr_dec_output), dim=2)
            attn_hidden = self.drop1(self.attn1(attn_input))
            attn_hidden = self.drop2(self.attn2(attn_hidden))
            attn_weights = torch.squeeze(self.attn3(attn_hidden), dim=2)
            attn_weights = nn.functional.softmax(attn_weights).reshape(attn_weights.shape[0], attn_weights.shape[1], 1)

            context_vector = torch.mul(enc_output, attn_weights)
            context_vector = torch.sum(context_vector, 1)
            context_vectors[:, i, :] = context_vector
            
        attn_vector = torch.cat((dec_output, context_vectors), dim=2)
        output = self.output_layer(attn_vector)

        return output


class LSTMAttnEncoderDecoder(nn.Module):
    def __init__(self, n_classes, n_emb_dim, n_hidden, n_lstm_units):
        super(LSTMAttnEncoderDecoder, self).__init__()
        self.encoder = Encoder(n_classes, n_emb_dim, n_hidden, n_lstm_units)
        self.decoder = AttnDecoder(n_classes, n_emb_dim, n_hidden, n_lstm_units)

    def forward(self, x, y):
        enc_output, enc_states = self.encoder.forward(x)
        output = self.decoder.forward(y, enc_output, enc_states)

        return output
