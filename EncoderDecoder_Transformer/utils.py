#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
##########################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import sys
import nltk

nltk.download('punkt')


class progressBar:

    def __init__(self, barWidth=50):
        self.barWidth = barWidth
        self.period = None

    def start(self, count):
        self.item = 0
        self.period = int(count / self.barWidth)
        sys.stdout.write("[" + (" " * self.barWidth) + "]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth + 1))

    def tick(self):
        if self.item > 0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1

    def stop(self):
        sys.stdout.write("]\n")


def readCorpus(fileName):
    ### Чете файл от изречения разделени с нов ред `\n`.
    ### fileName е името на файла, съдържащ корпуса
    ### връща списък от изречения, като всяко изречение е списък от думи
    print('Loading file:', fileName)
    with open(fileName, 'r', encoding='utf-8') as fileName:
        return [nltk.word_tokenize(line) for line in fileName]


def getDictionary(corpus,
                  startToken,
                  endToken,
                  unkToken,
                  padToken,
                  wordCountThreshold=2):
    dictionary = {}
    for s in corpus:
        for w in s:
            if w in dictionary: dictionary[w] += 1
            else: dictionary[w] = 1

    words = [startToken, endToken, unkToken, padToken] + [
        w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold
    ]
    return {w: i for i, w in enumerate(words)}


def prepareData(source_file_name, target_file_name, source_dev_file_name,
                target_dev_file_name, source_test_file_name,
                target_test_file_name, start_token, end_token, unk_token,
                pad_token):

    source_corpus = readCorpus(source_file_name)
    target_corpus = readCorpus(target_file_name)
    source_word2ind = getDictionary(source_corpus, start_token, end_token,
                                    unk_token, pad_token)
    target_word2ind = getDictionary(target_corpus, start_token, end_token,
                                    unk_token, pad_token)

    target_corpus = [[start_token] + s + [end_token] for s in target_corpus]

    source_dev = readCorpus(source_dev_file_name)
    target_dev = readCorpus(target_dev_file_name)

    target_dev = [[start_token] + s + [end_token] for s in target_dev]

    source_test = readCorpus(source_test_file_name)
    target_test = readCorpus(target_test_file_name)

    target_test = [[start_token] + s + [end_token] for s in target_test]

    print('Corpus loading completed.')

    return source_corpus, source_word2ind, target_corpus, target_word2ind, source_dev, target_dev, source_test, target_test
