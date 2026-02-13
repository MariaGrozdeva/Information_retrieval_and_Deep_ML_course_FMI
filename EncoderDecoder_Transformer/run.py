#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *


def perplexity(nmt, sourceTest, targetTest, batchSize):
    testSize = len(sourceTest)
    H = 0.
    c = 0
    for b in range(0, testSize, batchSize):
        sourceBatch = sourceTest[b:min(b + batchSize, testSize)]
        targetBatch = targetTest[b:min(b + batchSize, testSize)]
        l = sum(len(s) - 1 for s in targetBatch)
        c += l
        with torch.no_grad():
            H += l * nmt(sourceBatch, targetBatch)
    return math.exp(H / c)


if len(sys.argv) > 1 and sys.argv[1] == 'prepare':

    source_corpus, source_word2ind, target_corpus, target_word2ind, source_dev, target_dev, source_test, target_test = utils.prepareData(
        sourceFileName, targetFileName, sourceDevFileName, targetDevFileName,
        sourceTestFileName, targetTestFileName, startToken, endToken, unkToken,
        padToken)

    pickle.dump((source_corpus, target_corpus, source_dev, target_dev,
                 source_test, target_test), open(corpusFileName, 'wb'))
    pickle.dump((source_word2ind, target_word2ind), open(wordsFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv) > 1 and (sys.argv[1] == 'train'
                          or sys.argv[1] == 'extratrain'):

    source_corpus, target_corpus, source_dev, target_dev, source_test, target_test = pickle.load(
        open(corpusFileName, 'rb'))
    source_word2ind, target_word2ind = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(n_layers, n_heads, d_model, d_keys, d_values,
                              d_ff, seq_len, dropout, source_word2ind,
                              target_word2ind, startToken, endToken, unkToken,
                              padToken).to(device)

    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)

    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (iter, bestPerplexity, learning_rate,
         osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(source_corpus), dtype='int32')
    nmt.train()
    beginTime = time.time()

    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()

        for b in range(0, len(idx), batchSize):
            #############################################################################
            ### Може да се наложи да се променя скоростта на спускане learning_rate в зависимост от итерацията
            #############################################################################
            iter += 1

            batch_idx = idx[b:min(b + batchSize, len(idx))]
            sourceBatch = [source_corpus[i] for i in batch_idx]
            targetBatch = [target_corpus[i] for i in batch_idx]

            st = sorted(list(zip(sourceBatch, targetBatch)),
                        key=lambda e: len(e[0]),
                        reverse=True)
            (sourceBatch, targetBatch) = tuple(zip(*st))

            words += sum(len(s) - 1 for s in targetBatch)

            H = nmt(sourceBatch, targetBatch)
            optimizer.zero_grad()
            H.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(),
                                                       clip_grad)
            optimizer.step()

            if iter % log_every == 0:
                print("Iteration:", iter, "Epoch:", epoch + 1, '/', maxEpochs,
                      ", Batch:", b // batchSize + 1, '/',
                      len(idx) // batchSize + 1, ", loss: ", H.item(),
                      "words/sec:", words / (time.time() - trainTime),
                      "time elapsed:", (time.time() - beginTime))
                trainTime = time.time()
                words = 0

            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, source_dev, target_dev,
                                               batchSize)
                nmt.train()
                print('Current model perplexity: ', currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((iter, bestPerplexity, learning_rate,
                                optimizer.state_dict()),
                               modelFileName + '.optim')

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, source_dev, target_dev, batchSize)
    print('Last model perplexity: ', currentPerplexity)

    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save(
            (iter, bestPerplexity, learning_rate, optimizer.state_dict()),
            modelFileName + '.optim')

if len(sys.argv) > 3 and sys.argv[1] == 'perplexity':

    source_word2ind, target_word2ind = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(n_layers, n_heads, d_model, d_keys, d_values,
                              d_ff, seq_len, dropout, source_word2ind,
                              target_word2ind, startToken, endToken, unkToken,
                              padToken).to(device)
    nmt.load(modelFileName)
    nmt.eval()

    sourceTest = utils.readCorpus(sys.argv[2])
    targetTest = utils.readCorpus(sys.argv[3])
    targetTest = [[startToken] + s + [endToken] for s in targetTest]

    print('Model perplexity: ',
          perplexity(nmt, sourceTest, targetTest, batchSize))

if len(sys.argv) > 3 and sys.argv[1] == 'translate':

    source_word2ind, target_word2ind = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(n_layers, n_heads, d_model, d_keys, d_values,
                              d_ff, seq_len, dropout, source_word2ind,
                              target_word2ind, startToken, endToken, unkToken,
                              padToken).to(device)
    nmt.load(modelFileName)
    nmt.eval()

    sourceTest_tokens = utils.readCorpus(sys.argv[2])

    file = open(sys.argv[3], 'w', encoding='utf-8')
    pb = utils.progressBar()
    pb.start(len(sourceTest_tokens))

    for s in sourceTest_tokens:
        file.write(' '.join(nmt.generate(s, 1000)) + '\n')
        pb.tick()

    pb.stop()
    file.close()

if len(sys.argv) > 3 and sys.argv[1] == 'bleu':
    ref = [[s] for s in utils.readCorpus(sys.argv[2])]
    hyp = utils.readCorpus(sys.argv[3])

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
