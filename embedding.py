### t-Sne:
from rnn_seq_2 import SequenceModel
from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd
import numpy as np

seq_model = SequenceModel()
seq_model.load()

print('Experiment')

xdata = [x for x in range(0,1000)]
squares = [x*x for x in range(1,40) if x*x < 1000]
pow2 = [2**x for x in range(1,11) if 2**x < 1000]


embedded_data = seq_model.embed_int(xdata)
tsne = manifold.TSNE(n_components=2, init='pca', random_state=42)

#import ipdb;ipdb.set_trace()

#normalized = (embedded_data - np.mean(embedded_data, axis=0)) / np.std(embedded_data, axis=0)
#X_tsne = tsne.fit_transform(normalized)

X_tsne1000 = tsne.fit_transform(embedded_data)
X_tsne100 = tsne.fit_transform(embedded_data[:100])

def scatter_tsne(X_tsne, nums, c):
    plt.scatter(X_tsne[nums][:,0], X_tsne[nums][:,1], c=c)

def plot_tsne(X_tsne, nums, fmt='o'):
    plt.plot(X_tsne[nums][:,0], X_tsne[nums][:,1], fmt)

def plot_label(pt, label, fc='red'):
    plt.annotate(label, pt, color=fc, path_effects=[PathEffects.withStroke(linewidth=2, foreground='w')])

def plot_num_embeddings():
    """
    Plot colors from 1 to 100
    """
    colors = [x%10 for x in range(100)]
    scatter_tsne(X_tsne100, xdata[:100], colors)
    for x in range(100):
        plot_label(X_tsne100[x], x)
    plt.savefig('img/100-embedding-ones.png')
    plt.clf()


    plot_tsne(X_tsne100, xdata[:100], '-o')
    plt.savefig('img/100-embedding-seq.png')
    plt.clf()

    """
    Plot colors from 1 to 1000
    """
    plot_tsne(X_tsne1000, xdata, '-o')
    plt.savefig('img/1000-embedding.png')
    plt.clf()

    """
    Plot colors by %
    """
    colors = [x % 10 for x in range(1000)]
    scatter_tsne(X_tsne1000, xdata, colors)
    plt.savefig('img/1000-embedding-last-digit.png')
    plt.clf()


with open('data/test-data.txt') as f:
    def split_row(row):
        cells = row.strip().split('\t')
        return cells[0], list(map(int, cells[1:]))
    test_data = list(map(split_row, f.readlines()))


seq_paths = []
seq_embedded = []
for line in test_data[:300]:
    vec = seq_model.embed_seq(line[1])
    seq_embedded.append(vec)
    seq_paths.append(line[0])


label_data = [
('x', [1, 2, 3, 4, 5, 6, 7, 8]),
('x+1', [2,3,4,5,6,7,8,9]),
('x+1334', [1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342]),
('[10,3,4] repeat', [10,3,4,10,3,4,10,3]),
('[54,96,87, 23] repeat', [54, 96, 87, 23, 54, 96, 87, 23]),
('x^2', [1, 2, 4, 8, 16, 32, 64, 128]),
('3x^2', [3, 6, 12, 24, 48, 48, 192, 384]),
('100-x', [100, 99, 98, 97, 96, 95, 94, 93]),
('1000-x', [1000, 999, 998, 997, 996, 995, 994, 993]),
('8x', [1, 8, 16, 24, 32, 40, 48, 56]),
('500 + x', [501, 502, 503, 504, 505, 506, 507, 508]),
('601 - x', [600, 599, 598, 597, 596, 595, 594, 593]),
('9000 - 3x', [8997, 8994, 8991, 8988, 8985, 8982, 8979, 8976]),
('95234 - x', [95234, 95233, 95232, 95231, 95230, 95229, 95228, 95227]),
('51343 - 2x', [51343, 51341, 51339, 51337, 51335, 51333, 51331, 51329]),
('120000 + x', [120001, 120002, 120003, 120004, 120005, 120006, 120007, 120008]),
('215000 - x', [215499, 215498, 215497, 215496, 215495, 215494, 215493, 215493]),
]

label_embed = []
label_pat = []
for pat, seq in label_data:
    label_embed.append(seq_model.embed_seq(seq))
    label_pat.append(pat)


X_tsne_seq = tsne.fit_transform(np.matrix(seq_embedded + label_embed))
xdata = [x for x in range(00, len(X_tsne_seq))]

label_tsne_seq = X_tsne_seq[len(seq_embedded):]


plot_tsne(X_tsne_seq, xdata)
# for x in range(0, 300):
#    plot_label(X_tsne_seq[x], seq_paths[x], 'black')
for x in range(len(label_tsne_seq)):
    plot_label(X_tsne_seq[x+len(seq_embedded)], label_pat[x], 'red')
plt.savefig('img/tsne_seqs.png')

# for x in range(10):
#     plot_label(X_tsne, x)
#     plot_label(X_tsne, x*10)

plot_num_embeddings()
