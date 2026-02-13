#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2024/2025
#############################################################################
###
### Домашно задание 3
###
#############################################################################

import numpy as np
import torch


def generateText(model,
                 char2id,
                 auth,
                 startSentence,
                 limit=1000,
                 temperature=1.):
    # model е инстанция на обучен LSTMLanguageModelPack обект
    # char2id е речник за символите, връщащ съответните индекси
    # startSentence е началния низ стартиращ със символа за начало '{'
    # limit е горна граница за дължината на поемата
    # temperature е температурата за промяна на разпределението за следващ символ

    result = startSentence[1:]

    #############################################################################
    ###  Тук следва да се имплементира генерацията на текста
    #############################################################################
    #### Начало на Вашия код.

    device = next(model.parameters()).device
    model.eval()

    if auth is None:
        auth = '(неизвестен автор)'
    auth_id = model.auth2id.get(auth, 0)

    vocab_size = len(char2id)
    id2char = [""] * vocab_size
    for ch, i in char2id.items():
        id2char[i] = ch

    def sample_from_probs(probs_torch):
        p = probs_torch.detach().cpu().numpy()
        p = p / p.sum()
        return int(np.random.choice(len(p), p=p))

    with torch.no_grad():
        A = torch.tensor([auth_id], dtype=torch.long, device=device)
        auth_vec = model.authEmbed(A)

        h = model.auth2h0(auth_vec).view(model.lstm_layers, 1,
                                         model.hidden_size).contiguous()
        c = model.auth2c0(auth_vec).view(model.lstm_layers, 1,
                                         model.hidden_size).contiguous()

        for ch in startSentence:
            x_id = char2id.get(ch, model.unkTokenIdx)
            x = torch.tensor([[x_id]], dtype=torch.long, device=device)
            emb = model.embed(x)
            out, (h, c) = model.lstm(emb, (h, c))

        for _ in range(max(0, limit - len(result))):
            hidden = out[-1, 0, :]
            logits = model.projection(hidden)

            tau = float(temperature)
            if tau <= 0:
                tau = 1e-6

            probs = torch.nn.functional.softmax(logits / tau, dim=0)
            next_id = sample_from_probs(probs)
            next_ch = id2char[next_id]

            if next_ch == '}':
                break

            result += next_ch

            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            emb = model.embed(x)
            out, (h, c) = model.lstm(emb, (h, c))

    #### Край на Вашия код
    #############################################################################

    return result
