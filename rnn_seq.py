import numpy as np
import tensorflow as tf
from seqs import generate_seq

PAD = 0
EOS = 1

tf.reset_default_graph()

class SequenceModel:
    def __init__(self, lstm_size=256, num_layers=2):
        self.input_seq = tf.placeholder(tf.int32, shape=(None, None, None))
        self.input_seq_2 = tf.placeholder(tf.int32, shape=(None, None, None))
        batches, seq_len, digits = tf.unstack(tf.shape(self.input_seq))

        self.encoder_inputs = tf.transpose(tf.reshape(self.input_seq, (batches * seq_len, digits)))

        # total number of digits 0-9, <eos>, 0-padding
        self.vocab_size = 12

        # embedding size of digits
        self.input_embedding_size = 20

        # encoded dimensions of number
        self.encoder_hidden_units = 20
        self.decoder_hidden_units = self.encoder_hidden_units
        self.encode_size = self.encoder_hidden_units * 2

        encoder_max_time, batch_size = tf.unstack(tf.shape(self.encoder_inputs))

        # add <eos> to end of each sequence
        encoder_inputs_t = tf.concat([tf.transpose(self.encoder_inputs), tf.zeros([batch_size, 1], dtype=tf.int32)], 1)

        # find the lengths of each input sequence by getting the index of <eos>
        def index1d(t):
            return tf.cast(tf.reduce_min(tf.where(tf.equal(t, 0))), tf.int32)
        self.encoder_inputs_length = tf.transpose(tf.map_fn(index1d, encoder_inputs_t, dtype=tf.int32))

        # output sequence digits
        self.output_seq = tf.placeholder(tf.int32, shape=(None, None, None))
        _, _, output_digits = tf.unstack(tf.shape(self.output_seq))

        # flatten output target digits to [batches*seq len, output digits]
        self.decoder_targets = tf.transpose(tf.reshape(self.output_seq, (batches*seq_len, output_digits)))

        # max length for decoding numbers
        self.decoder_lengths = output_digits

        # E: embedding matrix for each digit
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=tf.float32)

        # E[Xij]
        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        """ Encoder """

        # Re
        self.encoder_cell = tf.contrib.rnn.LSTMCell(self.encoder_hidden_units)

        # run input sequence through encoder
        encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length,
            dtype=tf.float32, time_major=True,
        )

        del encoder_outputs

        # ui
        self.encoded_state = tf.concat([self.encoder_final_state.c, self.encoder_final_state.h], axis=1)

        """ Nested LSTM """
        self.encoder_outputs_batch = tf.reshape(self.encoded_state, (batches, seq_len, self.encode_size))

        # number of units in the cell
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.out_size = self.encode_size

        self.lstm_last_state = np.zeros((self.num_layers*2*self.lstm_size,))
        self.lstm_init_value = tf.placeholder(tf.float32, shape=(None, self.num_layers*2*self.lstm_size), name="lstm_init_value")

        # Rn
        self.lstm_cells = [ tf.contrib.rnn.BasicLSTMCell(self.lstm_size, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]
        self.lstm = tf.contrib.rnn.MultiRNNCell(self.lstm_cells, state_is_tuple=False)

        # Run through nested RNN
        outputs, self.nested_state = tf.nn.dynamic_rnn(self.lstm, self.encoder_outputs_batch, initial_state=self.lstm_init_value, dtype=tf.float32)

        """ Decoder """
        output_flat = tf.reshape(outputs, [-1, self.lstm_size])
        self.rnn_W = tf.Variable(tf.random_uniform([self.lstm_size, self.out_size], -1, 1), dtype=tf.float32)
        self.rnn_b = tf.Variable(tf.zeros([self.out_size]), dtype=tf.float32)
        self.decoder_inputs = tf.add(tf.matmul(output_flat, self.rnn_W), self.rnn_b)
        #self.decoder_inputs = output_flat
        enc_c, enc_h = tf.split(self.decoder_inputs, 2, axis=1)
        self.decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c=enc_c, h=enc_h)

        batch_size, _ = tf.unstack(tf.shape(output_flat))

        # Rd
        self.decoder_cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_units, reuse=True)

        self.rnn_W2 = tf.Variable(tf.random_uniform([self.decoder_hidden_units, self.vocab_size], -1, 1), dtype=tf.float32)
        self.rnn_b2 = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32)

        self.eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
        self.pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

        self.eos_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.eos_time_slice)
        self.pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.pad_time_slice)

        def digit_fn_rnn(initial_state):
            def loop_fn_initial():
                initial_elements_finished = (0 >= 13)  # all False at the initial step
                initial_input = self.eos_step_embedded
                initial_cell_state = initial_state
                initial_cell_output = None
                initial_loop_state = None  # we don't need to pass any additional information
                return (initial_elements_finished,
                        initial_input,
                        initial_cell_state,
                        initial_cell_output,
                        initial_loop_state)

            def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

                def get_next_input():
                    output_logits = tf.add(tf.matmul(previous_output, self.rnn_W2), self.rnn_b2)
                    prediction = tf.argmax(output_logits, axis=1)
                    next_input = tf.nn.embedding_lookup(self.embeddings, prediction)
                    return next_input
                
                elements_finished = (time >= self.decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                              # defining if corresponding sequence has ended

                finished = tf.reduce_all(elements_finished) # -> boolean scalar
                input = tf.cond(finished, lambda: self.pad_step_embedded, get_next_input)
                state = previous_state
                output = previous_output
                loop_state = None

                return (elements_finished, 
                        input,
                        state,
                        output,
                        loop_state)

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:    # time == 0
                    assert previous_output is None and previous_state is None
                    return loop_fn_initial()
                else:
                    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
            return loop_fn

        output_loop_fn = digit_fn_rnn(self.decoder_initial_state)
        self.decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, output_loop_fn)
        self.decoder_outputs = self.decoder_outputs_ta.stack()

        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(self.decoder_outputs))
        decoder_outputs_flat = tf.reshape(self.decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.rnn_W2), self.rnn_b2)
        #decoder_logits_flat = tf.reshape(self.decoder_outputs, (-1, decoder_dim))
        self.decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.vocab_size))

        self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
            logits=self.decoder_logits,
        )
        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)

        # train
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def load(self, model_fi="saved/model.ckpt"):
        self.saver.restore(self.sess, model_fi)

    def train_batch(self, xbatch, ybatch):
        batches = xbatch.shape[0]
        init_value = np.zeros((batches, self.num_layers*2*self.lstm_size))

        loss, train_op = self.sess.run([
            self.loss, self.train_op], feed_dict={
            self.input_seq: xbatch,
            self.output_seq: ybatch,
            self.lstm_init_value: init_value,
            })

        return loss, train_op

    def _predict(self, xbatch0):
        seq_len = xbatch0.shape[0]
        init_value = np.zeros((1, self.num_layers * 2 * self.lstm_size))

        return self.sess.run(self.decoder_prediction, feed_dict={
            self.input_seq: np.array([xbatch0]),
            self.lstm_init_value: init_value,
            self.decoder_lengths: 13
            })

    def _encode(self, seq):
        init_value = np.zeros((1, self.num_layers * 2 * self.lstm_size))
        return self.sess.run(self.encoded_state, feed_dict={
            self.input_seq: np.array([seq]),
            self.lstm_init_value: init_value,
            self.decoder_lengths: 13
            })

    def _encode_seq(self, seq):
        init_value = np.zeros((1, self.num_layers * 2 * self.lstm_size))
        return self.sess.run(self.nested_state, feed_dict={
            self.input_seq: np.array([seq]),
            self.lstm_init_value: init_value,
            self.decoder_lengths: 13
            })

    def predict(self, ints, max_digits=10):
        seq = [int_to_seq(x) for x in ints]
        inp_seq = np.zeros(shape=[len(seq), max_digits])
        for i in range(len(seq)):
            for j in range(len(seq[i])):
                inp_seq[i][j] = seq[i][j]

        pred = self._predict(inp_seq)
        return [seq_to_int(list(s)) for s in pred.T]


    def embed_int(self, ints, max_digits=10):
        seq = [int_to_seq(x) for x in ints]
        nums = np.zeros(shape=[len(seq), max_digits])
        inp_seq = np.zeros(shape=[len(seq), max_digits])
        for i in range(len(seq)):
            for j in range(len(seq[i])):
                inp_seq[i][j] = seq[i][j]

        return self._encode(inp_seq)

    def embed_seq(self, ints, max_digits=10):
        seq = [int_to_seq(x) for x in ints]
        nums = np.zeros(shape=[len(seq), max_digits])
        inp_seq = np.zeros(shape=[len(seq), max_digits])
        for i in range(len(seq)):
            for j in range(len(seq[i])):
                inp_seq[i][j] = seq[i][j]

        return self._encode_seq(inp_seq)[0]


def int_to_seq(num):
    assert isinstance(num, int) or isinstance(num, np.int32) or isinstance(num, np.int64)
    seq = [ord(x)-ord('0')+2 for x in str(num)]
    seq.append(0)
    return seq

def seq_to_int(seq):
    assert isinstance(seq, list)

    eos = 0
    while eos < len(seq) and seq[eos] not in [0,1]:
        eos += 1

    return sum([int(10**d * (x-2)) for d, x in enumerate(seq[:eos][::-1])])


def batch_seq(seqs, seq_len=8, max_digits=10):
    batch_size = len(seqs)

    input_seqs = []
    outputs_seqs = []

    for ints in seqs:
        seq = [int_to_seq(x) for x in ints]
        input_seqs.append(seq[:-1])
        output_seq = [(num[:-1]) + [EOS] + [PAD] * 2 for num in seq[1:]]
        outputs_seqs.append(output_seq)

    batch_input = np.zeros(shape=[batch_size, seq_len, max_digits+1])
    batch_output = np.zeros(shape=[batch_size, seq_len, max_digits+3])

    for i in range(len(input_seqs)):
        for j in range(len(input_seqs[i])):
            for k in range(len(input_seqs[i][j])):
                batch_input[i][j][k] = input_seqs[i][j][k]
    for i in range(len(outputs_seqs)):
        for j in range(len(outputs_seqs[i])):
            for k in range(len(outputs_seqs[i][j])):
                batch_output[i][j][k] = outputs_seqs[i][j][k]
    return batch_input, batch_output


def next_seqs(batch_size=100, seq_len=8, max_digits=10):
    seqs = []
    seq_strs = []
    for b in range(batch_size):
        ints = []
        seq_str = ''

        while True:
            ints, seq_str = generate_seq(seq_len+1)
            if max(ints) < 10000000 and min(ints) > 0:
                break
        seqs.append(ints)
        seq_strs.append(seq_str)
    return seqs, seq_strs


with open('data/test-data.txt') as f:
    def split_row(row):
        cells = row.strip().split('\t')
        return cells[0], list(map(int, cells[1:]))
    test_data = list(map(split_row, f.readlines()))


def eval_test_data(seq_model):
    correct = 0
    for line in test_data:
        pred = seq_model.predict(line[1])
        if pred[-2] == line[1][-1]:
            correct += 1
        else:
            print(line[0])
            print(pred)
            print(line[1])
    return correct / len(test_data)


def main():
    #### Training #####
    for i in range(50001):
        seqs, seq_str = next_seqs()
        #print(seqs)
        xbatch, ybatch = batch_seq(seqs)
     
        if i % 100 == 0:
            pred = seq_model._predict(xbatch[0])
            print('Input', [seq_to_int(list(s)) for s in xbatch[0]])
            print('Output', [seq_to_int(list(s)) for s in ybatch[0]])
            print('Pred', [seq_to_int(list(s)) for s in pred.T])
            print('Pat', seq_str[0])

            # print('Test acc', eval_test_data(seq_model))

        loss, summary = seq_model.train_batch(xbatch, ybatch)
            
        if i % 100 == 0:
            # summary_writer.add_summary(summary, i)
            print(i, loss)
            seq_model.saver.save(seq_model.sess, "saved/model.ckpt")


if __name__ == '__main__':

    seq_model = SequenceModel()
    seq_model.load()

    # print(eval_test_data(seq_model))

    main()

    





### Experiment 2: Predict 1000 digits of PI  err per digit < 0.1
# def make_pi():
#     q, r, t, k, m, x = 1, 0, 1, 1, 3, 3
#     for j in range(10000):
#         if 4 * q + r - t < m * t:
#             yield m
#             q, r, t, k, m, x = 10*q, 10*(r-m*t), t, k, (10*(3*q+r))//t - 10*m, x
#         else:
#             q, r, t, k, m, x = q*k, (2*q+r)*x, t*x, k+1, (q*(7*k+2)+r*x)//(t*x), x+2

# pi_digits = [x for x in make_pi()]
# print('Experiment 2')
# print('PI', pi_digits[:500])
# print('Predicted: ',predict(pi_digits[:500], 6))

# ### Experiment 3: Train on first 500 digits of PI and predict next 500 digits
# pi_digits_batched = [pi_digits[i:i+50] for i in range(0, 10000, 50)]
# xbatch, ybatch = batch_seq(pi_digits_batched, seq_len=50)
# seq_model.train_batch(xbatch, ybatch)
# print('PI', ''.join(str(x) for x in pi_digits[:500]))
# print('Predicted', ''.join(str(x) for x in predict(pi_digits[:500], 6)))


### Experiment 3: Same for e

### Experiment 4: Primes






