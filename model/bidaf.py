import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basic_layers import Linear
from model.basic_layers import LSTM
from model.basic_layers import get_loss_mask


class BidafModel(nn.Module):

    def __init__(self, params, data, padding_idx=0):
        super(BidafModel, self).__init__()

        self.params = params
        self.hn_dims = params['dim_embeddings'] + params['out_channel_dims']

        # 1. Character Embedding Layer.
        # init char embeddings to random vectors
        self.char_emb = nn.Embedding(len(data.CHAR.vocab), params['char_out_size'], padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.char_conv = nn.Conv1d(in_channels=params['char_out_size'], out_channels=params['out_channel_dims'],
                                   kernel_size=params['filter_heights'])

        # 2. Word Embedding Layer.
        # word embeddings are not fine-tuned
        self.word_embed = nn.Embedding.from_pretrained(data.WORD.vocab.vectors, freeze=True)

        # highway network
        self.hn_transform = nn.ModuleList()
        self.hn_gate = nn.ModuleList()
        for i in range(params['highway_network_layers']):
            transform_layer = nn.Sequential(Linear(self.hn_dims, self.hn_dims), nn.ReLU())
            gate_layer = nn.Sequential(Linear(self.hn_dims, self.hn_dims), nn.Sigmoid())

            self.hn_transform.append(transform_layer)
            self.hn_gate.append(gate_layer)

        # 3. Contextual  Embedding  Layer.
        self.contextual_LSTM = LSTM(input_size=self.hn_dims,
                                    hidden_size=params['lstm_hidden_size'],
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=params['dropout'])

        # 4. Attention Flow Layer.
        # self.attention_layer = Linear(params['out_channel_dims'] * 6, 1)

        self.attention_weight_c = Linear(params['out_channel_dims'] * 2, 1)
        self.attention_weight_q = Linear(params['out_channel_dims'] * 2, 1)
        self.attention_weight_cq = Linear(params['out_channel_dims'] * 2, 1)

        # 5. Modeling Layer.
        self.modeling_LSTM1 = LSTM(input_size=8 * params['lstm_hidden_size'],
                                   hidden_size=params['lstm_hidden_size'],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=params['dropout'])

        self.modeling_LSTM2 = LSTM(input_size=2 * params['lstm_hidden_size'],
                                   hidden_size=params['lstm_hidden_size'],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=params['dropout'])

        # 6. Output Layer.
        self.p1_weight_g = Linear(8 * params['lstm_hidden_size'], 1, dropout=params['dropout'])
        self.p1_weight_m = Linear(2 * params['lstm_hidden_size'], 1, dropout=params['dropout'])
        self.p2_weight_g = Linear(8 * params['lstm_hidden_size'], 1, dropout=params['dropout'])
        self.p2_weight_m = Linear(2 * params['lstm_hidden_size'], 1, dropout=params['dropout'])

        self.output_LSTM = LSTM(input_size=params['lstm_hidden_size'] * 2,
                                hidden_size=params['lstm_hidden_size'],
                                bidirectional=True,
                                batch_first=True,
                                dropout=params['dropout'])

    def embed_chars(self, batch_char):
        """
        batch_char: (batch, seq_len, char_len)
        """
        batch_size = batch_char.size(0)

        # batch, seq_len, char_len, char_emb_dim
        x = self.char_emb(batch_char)

        # batch*seq_len, char_emb_dim, char_len
        x = x.view(-1, x.size(3), x.size(2))

        # batch*seq_len, out_channel_dims, conv_len
        x = self.char_conv(x)

        # batch*seq_len, out_channel_dims
        x = F.max_pool1d(x, x.size(2)).squeeze()

        # batch, seq_len, char_out_size
        x = x.view(batch_size, -1, self.params['char_out_size'])

        return x

    def highway_network(self, x):
        for i in range(self.params['highway_network_layers']):
            t = self.hn_transform[i](x)
            g = self.hn_gate[i](x)
            x = g * t + (1 - g) * x

        return x

    def attention_flow_layer(self, c, q, c_lengths, q_lengths):
        c_seq_len = c.size(1)
        q_seq_len = q.size(1)


        q_mask = get_loss_mask(q_lengths).unsqueeze(1)

        s_hu = []
        for i in range(q_seq_len):
            # batch, 1, hidden_size * 2
            qi = q.select(1, i).unsqueeze(1)

            # batch, c_len, 1
            ci = self.attention_weight_cq(c * qi).squeeze()

            s_hu.append(ci)

        # batch, c_seq_len, q_seq_len
        s_hu = torch.stack(s_hu, dim=-1)

        # batch, c_seq_len, q_seq_len
        s = self.attention_weight_c(c).expand(-1, -1, q_seq_len) + self.attention_weight_q(q).permute(0, 2, 1).expand(-1, c_seq_len, -1) + s_hu

        a = F.softmax(s, dim=2)

        c2q = torch.bmm(a, q)

        # mask scores outside sentence len bounds to eliminate their influence on max function
        s_masked = s.masked_fill((1 - q_mask).byte(), -1e10)

        b = F.softmax(torch.max(s_masked, dim=2)[0], dim=1)

        q2c = torch.bmm(b.unsqueeze(1), c).squeeze()
        q2c = q2c.unsqueeze(1).expand(-1, c_seq_len, -1)

        g = torch.cat([c, c2q, c * c2q, c * q2c], dim=2)
        return g

    def output_layer(self, g, m, l):
        """
        g: (batch, c_len, hidden_size * 8)
        m: (batch, c_len ,hidden_size * 2)
        """
        # batch, c_len
        p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
        # batch, c_len, hidden_size * 2
        m2 = self.output_LSTM(m, l)
        # batch, c_len
        p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

        return p1, p2

    def forward(self, batch):
        """
        char dims: batch x seq_len x word_len
        """
        len_c = batch.c_word[1]
        len_q = batch.q_word[1]

        # 1
        char_embedded_c = self.embed_chars(batch.c_char)
        char_embedded_q = self.embed_chars(batch.q_char)

        # 2
        word_embedded_c = self.word_embed(batch.c_word[0])
        word_embedded_q = self.word_embed(batch.q_word[0])

        embedded_c = torch.cat([char_embedded_c, word_embedded_c], dim=2)
        embedded_q = torch.cat([char_embedded_q, word_embedded_q], dim=2)

        hn_embedded_c = self.highway_network(embedded_c)
        hn_embedded_q = self.highway_network(embedded_q)

        # 3
        contextual_c = self.contextual_LSTM(hn_embedded_c, len_c)
        contextual_q = self.contextual_LSTM(hn_embedded_q, len_q)

        # 4
        G = self.attention_flow_layer(contextual_c, contextual_q, len_c, len_q)

        # 5
        M = self.modeling_LSTM1(G, len_c)
        M = self.modeling_LSTM2(M, len_c)

        # 6
        p1, p2 = self.output_layer(G, M, len_c)

        return p1, p2
