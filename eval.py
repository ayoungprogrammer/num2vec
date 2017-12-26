from rnn_seq_2 import SequenceModel
import tensorflow as tf

###### Experiments #######
seq_model = SequenceModel()
seq_model.load()

### Experiment 1: What is the generalization of prediction? (i.e. can it predict sequence it hasn't seen?
print('Experiment 1')
seqs = [[43424, 43425, 43426, 43427, 43428],
[3,4,3,4,3,4,3,4,3,4],
[3,5,9,17,33,65,129],
[10, 9, 8, 7, 6, 5, 4, 3],
[1000001, 1000002, 1000003, 1000004],
[6, 5, 4, 3, 2, 1, 0],
[13, 12, 11, 10, 9, 8],]
for seq in seqs:
    print('Seq', seq)
    print('Predicted', seq_model.predict(seq))