#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import torch
import torch.nn as nn

#################################################################
####  LSTM с пакетиране на партида
#################################################################


class LSTMLanguageModelPack(nn.Module):

    def preparePaddedBatch(self, source):
        device = next(self.parameters()).device
        m = max(len(s) for (a, s) in source)
        sents = [[self.word2ind.get(w, self.unkTokenIdx) for w in s]
                 for (a, s) in source]
        auths = [self.auth2id.get(a, 0) for (a, s) in source]
        sents_padded = [s + (m - len(s)) * [self.padTokenIdx] for s in sents]
        return torch.t(
            torch.tensor(sents_padded, dtype=torch.long,
                         device=device)), torch.tensor(auths,
                                                       dtype=torch.long,
                                                       device=device)

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName, device):
        self.load_state_dict(torch.load(fileName, device))

    def __init__(self, embed_size, hidden_size, auth2id, word2ind, unkToken,
                 padToken, endToken, lstm_layers, dropout):
        super(LSTMLanguageModelPack, self).__init__()
        #############################################################################
        ###  Тук следва да се имплементира инициализацията на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавки за повече слоеве на РНН, влагане за автора и dropout
        #############################################################################
        #### Начало на Вашия код.

        self.auth2id = auth2id
        self.word2ind = word2ind

        self.unkTokenIdx = word2ind[unkToken]
        self.padTokenIdx = word2ind[padToken]
        self.endTokenIdx = word2ind[endToken]

        self.vocab_size = len(word2ind)
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

        self.embed = nn.Embedding(self.vocab_size,
                                  embed_size,
                                  padding_idx=self.padTokenIdx)

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=lstm_layers,
                            dropout=0.0,
                            bidirectional=False)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, self.vocab_size)
        self.authEmbed = nn.Embedding(len(auth2id), hidden_size)
        self.auth2h0 = nn.Linear(hidden_size, lstm_layers * hidden_size)
        self.auth2c0 = nn.Linear(hidden_size, lstm_layers * hidden_size)

        #### Край на Вашия код
        #############################################################################

    def forward(self, source):
        #############################################################################
        ###  Тук следва да се имплементира forward метода на обекта
        ###  За целта може да копирате съответния метод от програмата за упр. 13
        ###  като направите добавка за dropout и началните скрити вектори
        #############################################################################
        #### Начало на Вашия код.

        X, A = self.preparePaddedBatch(source)
        E = self.embed(X[:-1])

        lengths = [len(s) - 1 for (_, s) in source]
        auth_vec = self.authEmbed(A)

        h0 = self.auth2h0(auth_vec).view(self.lstm_layers, A.size(0),
                                         self.hidden_size).contiguous()
        c0 = self.auth2c0(auth_vec).view(self.lstm_layers, A.size(0),
                                         self.hidden_size).contiguous()

        packed = nn.utils.rnn.pack_padded_sequence(E,
                                                   lengths=lengths,
                                                   enforce_sorted=False)
        outputPacked, _ = self.lstm(packed, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(outputPacked)
        output = self.dropout(output)

        Z = self.projection(output.flatten(0, 1))
        Y = X[1:].flatten(0, 1)

        loss = nn.functional.cross_entropy(Z, Y, ignore_index=self.padTokenIdx)
        return loss

        #### Край на Вашия код
        #############################################################################
