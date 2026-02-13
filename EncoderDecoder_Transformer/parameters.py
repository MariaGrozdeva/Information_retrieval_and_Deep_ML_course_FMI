import torch
import nltk

nltk.download('punkt_tab')

sourceFileName = '/content/drive/MyDrive/en_bg_data/train.bg'
targetFileName = '/content/drive/MyDrive/en_bg_data/train.en'
sourceDevFileName = '/content/drive/MyDrive/en_bg_data/dev.bg'
targetDevFileName = '/content/drive/MyDrive/en_bg_data/dev.en'
sourceTestFileName = '/content/drive/MyDrive/en_bg_data/test.bg'
targetTestFileName = '/content/drive/MyDrive/en_bg_data/test.en'

corpusFileName = 'corpusData'
wordsFileName = 'wordsData'
modelFileName = 'NMTmodel'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

startToken = '<S>'
endToken = '</S>'
unkToken = '<UNK>'
padToken = '<PAD>'

learning_rate = 0.0001
batchSize = 32
clip_grad = 5.0

maxEpochs = 10
log_every = 10
test_every = 2000

n_layers = 3
n_heads = 8
d_model = 256
d_ff = 2048  # value from the paper
dropout = 0.1
seq_len = 200
d_keys = d_values = d_model // n_heads
