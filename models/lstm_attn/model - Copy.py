import torch.nn as nn
import torch
import os


class Encoder(nn.Module):
    def __init__(self, n_class, emb_dim, hidden_dim, lstm_units_enc, lstm_dropout, device):
        """
        Creates an Encoder model.

        Args:
            n_class (int): Number of classes/ Vocabulary size.
            emb_dim (int): Embeddings dimension.
            hidden_dim (int): LSTM hidden layers dimension.
            lstm_units_enc (int): Number of LSTM units for the encoder in a lstm layer.
            lstm_dropout (float): LSTM dropout.
            device : The device to run the model on.
        """

        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(n_class, emb_dim)
        # The encoder lstm layer is bidirectional.
        self.lstm = nn.LSTM(emb_dim, hidden_dim, lstm_units_enc, dropout=lstm_dropout, bidirectional=True,
                            batch_first=True)

        self.to(device)

    def forward(self, input):
        """
        Args:
            input (tensor): The input of the encoder. It must be a 2-D tensor of integers. Shape: [batch_size,
                seq_len_enc].

        Returns:
            A tuple containing the output and the states of the last LSTM layer. The states of the LSTM layer is also a
            tuple that contains the hidden and the cell state, respectively . Output shape: [batch_size, seq_len_enc,
            hidden_dim * 2]. Hidden/cell state shape: [lstm_units_enc*2, batch_size, hidden_dim].
        """

        # Creates the embeddings. [batch_size, seq_len] -> [batch_size, seq_len, emb_dim].
        embeddings = self.embedding(input)

        # Computes the output and the two states of the lstm layer. See function returns docs for details.
        output, states = self.lstm(embeddings)

        return output, states


class AttnDecoder(nn.Module):
    def __init__(self, n_class, emb_dim, input_size, hidden_dim, lstm_units_dec, lstm_dropout, dropout, device):
        """
        Creates an Decoder with attention.

        Args:
            n_class (int): Number of classes/ Vocabulary size.
            emb_dim (int): Embeddings dimension.
            input_size (int): Input size.
            hidden_dim (int): LSTM hidden layers dimension.
            lstm_units_dec (int): Number of LSTM units for the decoder in a lstm layer.
            lstm_dropout (float): LSTM dropout.
            dropout (float): The dropout in the attention layer.
            device : The device to run the model on.
        """

        super(AttnDecoder, self).__init__()

        self.embedding = nn.Embedding(n_class, emb_dim)

        # The attention layers. We use a hidden layer of dimension 4*hidden_dim.
        self.attn1 = nn.Linear(hidden_dim * lstm_units_dec + input_size, hidden_dim * 4)
        self.attn2 = nn.Linear(hidden_dim * 4, 1)
        self.dropout = nn.Dropout(dropout)

        # The decoder lstm layer is unidirectional.
        self.lstm = nn.LSTM(emb_dim + input_size, hidden_dim, lstm_units_dec, dropout=lstm_dropout, batch_first=True)

        self.output_linear = nn.Linear(hidden_dim, n_class)

        self.to(device)

    def _calculate_context_vector(self, state_h, enc_output):
        """
        This function calculates the context vector of the attention layer, given the hidden state and the encoder
        last lstm layer output.

        Args:
            state_h (tensor): The hidden state of the encoder's last LSTM layer(first token) or of the previous decoder
                state. Shape: [lstm_units_dec, batch_size, hidden_dim].
            enc_output (tensor): The output of the last LSTM encoder layer. Shape: [batch_size, seq_len_enc,
                hidden_dim * 2].

        Returns:
            The context vector. Shape: [batch_size, hidden_dim * lstm_units_dec]
        """

        batch_size = enc_output.shape[0]
        seq_len_enc = enc_output.shape[1]

        # Permutes the tensor dims to put the batch_size to the first position. Creates a 3-D sequence with a length of
        # one from the 2-D tensor and expands the sequence length to seq_len_enc. This makes the computation of
        # the cat between the enc_output and encoder much easier. [lstm_units_dec, batch_size, hidden_dim] ->
        # [batch_size, seq_len_enc, hidden_dim * lstm_units_dec].
        state_h = state_h.permute(1, 0, 2).reshape(batch_size, 1, -1).expand(-1, seq_len_enc, -1)

        # Concatenates the encoder output with the new hidden state over the third dimension.
        attn_input = torch.cat((enc_output, state_h), dim=2)

        # Calculates the attention weights.
        attn_hidden = self.attn1(attn_input)
        attn_hidden = self.dropout(attn_hidden)
        attn_output = self.attn2(attn_hidden)
        attn_weights = nn.functional.softmax(attn_output, dim=1)

        # Multiply the attention weights with the attn_weights.
        context_vector = torch.mul(enc_output, attn_weights)

        # Calculates the sum over the seq_len_enc.
        return torch.sum(context_vector, dim=1)

    def forward(self, input, enc_output, enc_states):
        """
        Args:
             input (tensor): The input of the decoder. Shape: [batch_size, seq_len_dec].
             enc_output (tensor): The output of the encoder last LSTM layer. Shape: [batch_size seq_len_enc,
                hidden_dim * 2].
             enc_states (tuple of tensors): The hidden and the cell states of the encoder last LSTM layer after being
                transformed by the linear layer. State shape: [lstm_units_dec, batch_size, hidden_dim].

        Returns:
            The output of the decoder, a tensor that contains a sequence of tokens with the dimension equal to
            vocabulary size. Shape: [batch_size, seq_len_dec, n_class].
        """
        batch_size = input.shape[0]
        seq_len_dec = input.shape[1]

        # Creates the embeddings. [batch_size, seq_len] -> [batch_size, seq_len, emb_dim].
        embeddings = self.embedding(input)

        # Calculates the context vector.
        context_vector = self._calculate_context_vector(enc_states[0], enc_output)

        # Concatenates the input of the <BOS> embedding with the context vector over the second dimensions. Transforms
        # the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] + [batch_size, hidden_dim *
        # lstm_units_dec] -> [batch_size, 1, emb_dim + hidden_dim * lstm_units_dec].
        lstm_input = torch.cat((embeddings[:, 0, :], context_vector), dim=1).reshape(batch_size, 1, -1)

        # Feeds the resulted first token to the lstm layer of the decoder. The initial state of the decoder is the
        # transformed state of the last LSTM layer of the encoder. [batch_size, seq_len_dec, hidden_dim],
        # [lstm_units_dec, batch_size, hidden_dim].
        dec_output, dec_states = self.lstm(lstm_input, enc_states)

        # Loop over the rest of tokens in the input seq_len_dec.
        for i in range(1, seq_len_dec):
            # Calculate the context vector at step i.
            context_vector = self._calculate_context_vector(dec_states[0], enc_output)

            # Concatenates the input of the i-th embedding with the corresponding  context vector over the second
            # dimensions. Transforms the 2-D tensor to 3-D sequence tensor with length 1. [batch_size, emb_dim] +
            # [batch_size, hidden_dim * lstm_units_dec] -> [batch_size, 1, emb_dim + hidden_dim * lstm_units_dec].
            lstm_input = torch.cat((embeddings[:, i, :], context_vector), dim=1).reshape(batch_size, 1, -1)

            # Calculates the i-th decoder output and state. We initialize the decoder state with (i-1)-th state.
            # [batch_size, seq_len_dec, hidden_dim], [lstm_units_dec, batch_size, hidden_dim].
            curr_dec_output, dec_states = self.lstm(lstm_input, dec_states)

            # Creates the final decoder LSTM output sequence by concatenating the i-th decoder LSTM output with the
            # previous decoder LSTM output sequence. [batch_size, i-1, hidden_dim] -> [batch_size, i, hidden_dim].
            dec_output = torch.cat((dec_output, curr_dec_output), dim=1)

        # Creates the output of the decoder. The last layer maps the output of the LSTM decoders to a tensor with the
        # last dimension equal to the vocabulary size. [batch_size, seq_len_dec, hidden_dim] -> [batch_size,
        # seq_len_dec, n_class].
        output = self.output_linear(dec_output)

        return output


class LSTMAttnEncoderDecoder(nn.Module):
    def __init__(self, n_class, emb_dim, hidden_dim, lstm_units_enc, lstm_units_dec, lstm_dropout, dropout):
        """
        Creates an Encoder-Decoder with attention.

        Args:
            n_class (int): Number of classes/ Vocabulary size.
            emb_dim (int): Embeddings dimension.
            hidden_dim (int): LSTM hidden layers dimension.
            lstm_units_enc (int): Number of LSTM units for the encoder in a lstm layer.
            lstm_units_dec (int): Number of LSTM units for the decoder in a lstm layer.
            lstm_dropout (float): LSTM dropout.
            dropout (float): The dropout in the attention layer.
            device : The device to run the model on.
        """

        super(LSTMAttnEncoderDecoder, self).__init__()

        if torch.cuda.is_available():
            print('Running on GPU.')
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            print('Running on CPU.')
            self.cuda = False
            self.device = torch.device('cpu')

        self.hidden_dim = hidden_dim        
        self.lstm_units_enc = lstm_units_enc
        self.lstm_units_dec = lstm_units_dec

        self.encoder = Encoder(n_class, emb_dim, hidden_dim, lstm_units_enc, lstm_dropout, self.device)
        self.decoder = AttnDecoder(n_class, emb_dim, hidden_dim*2, hidden_dim, lstm_units_dec, lstm_dropout, dropout,
                                   self.device)

        self.h_state_linear = nn.Linear(hidden_dim * lstm_units_enc * 2, hidden_dim * lstm_units_dec)
        self.c_state_linear = nn.Linear(hidden_dim * lstm_units_enc * 2, hidden_dim * lstm_units_dec)

        if self.cuda:
            self.to(self.device)

    def forward(self, x, y):
        """
        Args:
            x (tensor): The input of the decoder. Shape: [batch_size, seq_len_enc].
            y (tensor): The input of the decoder. Shape: [batch_size, seq_len_dec].

        Returns:
            The output of the Encoder-Decoder with attention. Shape: [batch_size, seq_len_dec, n_class].
        """

        batch_size = x.shape[0]

        # Calculates the output of the encoder
        enc_output, enc_states = self.encoder.forward(x)

        # Reshapes the shape of the hidden and cell state of the encoder last LSTM layer. Permutes the batch_size to
        # the first dimension, and reshapes them to a 2-D tensor. [lstm_units_enc * 2, batch_size, hidden_dim] ->
        # [batch_size, lstm_unites_enc * hidden_dim * 2].
        enc_states = (enc_states[0].permute(1, 0, 2).reshape(batch_size, -1),
                      enc_states[1].permute(1, 0, 2).reshape(batch_size, -1))

        # Transforms the hidden and the cell state of the encoder last lstm layer to correspond to the decoder lstm
        # states dimensions. [batch_size, lstm_units_enc * hidden_dim * 2] -> [batch_size, lstm_units_dec * hidden_dim].
        enc_states = (self.h_state_linear(enc_states[0]), self.c_state_linear(enc_states[1]))

        # Reshapes the states to have the correct shape for the decoder lstm states dimension. Reshape the states from
        # 2-D to 3-D sequence. Permutes the batch_size to the second dimension. [batch_size, lstm_units_dec *
        # hidden_dim] -> [lstm_units_dec, batch_size, hidden_size].
        enc_states = (enc_states[0].reshape(batch_size, self.lstm_units_dec, self.hidden_dim).permute(1, 0, 2),
                      enc_states[1].reshape(batch_size, self.lstm_units_dec, self.hidden_dim).permute(1, 0, 2))

        # Calculates the output of the decoder.
        output = self.decoder.forward(y, enc_output, enc_states)

        return output

    def load_checkpoint(self, folder, extension):
        filename = os.path.join(folder, "checkpoint." + extension)
        print("Loading model {} ...".format(filename))
        if not os.path.exists(filename):
            print("\t Model file not found, not loading anything!")
            return {}

        checkpoint = torch.load(filename)
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        return checkpoint["extra"]

    def save_checkpoint(self, folder, extension, extra={}):
        filename = os.path.join(folder, "checkpoint." + extension)
        checkpoint = {}
        checkpoint["encoder_state_dict"] = self.encoder.state_dict()
        checkpoint["decoder_state_dict"] = self.decoder.state_dict()
        checkpoint["extra"] = extra
        torch.save(checkpoint, filename)

    
        