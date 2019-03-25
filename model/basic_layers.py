import torch
import torch.nn as nn


def get_loss_mask(lens):
    max_len = lens.max().item()
    mask = torch.arange(max_len, device=lens.device, dtype=lens.dtype).expand(len(lens), max_len) < lens.unsqueeze(1)
    return mask


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   mask_fill_value: float = -1e32):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
        result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout_layer'):
            x = self.dropout_layer(x)
        x = self.linear(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        # self.init_params()
        self.dropout = nn.Dropout(p=dropout)

    def init_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l{}'.format(i)))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l{}'.format(i)))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), val=0)
            getattr(self.rnn, 'bias_hh_l{}'.format(i)).chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l{}_reverse'.format(i)))
                nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l{}_reverse'.format(i)))
                nn.init.constant_(getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)), val=0)
                nn.init.constant_(getattr(self.rnn, 'bias_ih_l{}_reverse'.format(i)), val=0)
                getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)).chunk(4)[1].fill_(1)

    def forward(self, x, x_len):
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, _ = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)

        return x
