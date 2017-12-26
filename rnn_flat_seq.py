import numpy as np
import tensorflow as tf
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

def index1d(t):
    return tf.cast(tf.reduce_min(tf.where(tf.equal(t, 0))), tf.int32)

class IntegerAutoEncoder():
    def __init__(self):

        self.vocab_size = 12
        self.input_embedding_size = 20

        self.encoder_hidden_units = 20
        self.decoder_hidden_units = self.encoder_hidden_units

        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        encoder_max_time, batch_size = tf.unstack(tf.shape(self.encoder_inputs))

        encoder_inputs_t = tf.concat([tf.transpose(self.encoder_inputs), tf.zeros([batch_size, 1], dtype=tf.int32)], 1)

        self.encoder_inputs_length = tf.transpose(tf.map_fn(index1d, encoder_inputs_t, dtype=tf.int32))
        #tf.placeholder(shape=(None, ), dtype=tf.int32, name='encoder_inputs_length')

        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=tf.float32)

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)

        self.encoder_cell = tf.contrib.rnn.LSTMCell(self.encoder_hidden_units)

        encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(
            self.encoder_cell, self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length,
            dtype=tf.float32, time_major=True,
        )

        del encoder_outputs

        self.encoded_state = tf.concat([self.encoder_final_state.c, self.encoder_final_state.h], axis=1)
        enc_c, enc_h = tf.split(self.encoded_state, 2, axis=1)
        self.decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c=enc_c, h=enc_h)

        batch_size, _ = tf.unstack(tf.shape(self.encoded_state))

        self.decoder_cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_units, reuse=True)

        self.rnn_W = tf.Variable(tf.random_uniform([self.decoder_hidden_units, self.vocab_size], -1, 1), dtype=tf.float32)
        self.rnn_b = tf.Variable(tf.zeros([self.vocab_size]), dtype=tf.float32)

        self.eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
        self.pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

        self.eos_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.eos_time_slice)
        self.pad_step_embedded = tf.nn.embedding_lookup(self.embeddings, self.pad_time_slice)

        self.decoder_lengths = self.encoder_inputs_length + 3
        # +2 additional steps, +1 leading <EOS> token for decoder inputs

        def loop_fn_initial():
            initial_elements_finished = (0 >= self.decoder_lengths)  # all False at the initial step
            initial_input = self.eos_step_embedded
            initial_cell_state = self.decoder_initial_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

            def get_next_input():
                output_logits = tf.add(tf.matmul(previous_output, self.rnn_W), self.rnn_b)
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

        self.decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn)
        self.decoder_outputs = self.decoder_outputs_ta.stack()

        decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(self.decoder_outputs))
        decoder_outputs_flat = tf.reshape(self.decoder_outputs, (-1, decoder_dim))
        decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.rnn_W), self.rnn_b)
        decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.vocab_size))

        self.decoder_prediction = tf.argmax(decoder_logits, 2)

        self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=self.vocab_size, dtype=tf.float32),
            logits=decoder_logits,
        )
        self.loss = tf.reduce_mean(self.stepwise_cross_entropy)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        sess.run(tf.global_variables_initializer())

    def _train(self, encoder_inputs, decoder_targets):
        return sess.run([self.train_op, self.loss], feed_dict={
            self.encoder_inputs: encoder_inputs,
            self.decoder_targets: decoder_targets,
        })

    def predict(self, encoder_inputs):
        return sess.run(self.decoder_prediction, feed_dict={self.encoder_inputs: encoder_inputs})

    def encode(self, encoder_inputs):
        return sess.run(self.encoded_state, feed_dict={self.encoder_inputs: encoder_inputs})

    def decode(self, encoded_state, max_decode_length=10):
        batch_size = encoded_state.shape[0]

        decoder_lengths = np.array([max_decode_length for i in range(batch_size)])
        return sess.run(self.decoder_prediction, feed_dict={self.encoded_state: encoded_state,
            self.decoder_lengths: decoder_lengths})

    def train(self, epochs=5001, batch_size=100, length=4):
        batches = helpers.random_sequences(length_from=1, length_to=length,
                                           vocab_lower=2, vocab_upper=12,
                                           batch_size=batch_size)

        print('head of the batch:')
        for seq in next(batches)[:10]:
            print(seq)

        def next_feed():
            batch = next(batches)
            encoder_inputs_, _ = helpers.batch(batch)
            decoder_targets_, _ = helpers.batch(
                [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
            )
            return encoder_inputs_, decoder_targets_,
            
        max_batches = 10001
        batches_in_epoch = 1000

        for batch in range(max_batches):
            encoder_inputs, decoder_targets = next_feed()

            if batch % batches_in_epoch == 0:
                print(encoder_inputs.T[0:2])
                dec = sess.run(self.decoder_prediction, feed_dict={self.encoder_inputs: encoder_inputs})
                print(dec.T[0:2])

            _, loss = self._train(encoder_inputs, decoder_targets)

            if batch % batches_in_epoch ==0:
                print(batch, loss)


def int_to_seq(num):
    assert isinstance(num, int) or isinstance(num, np.int32) or isinstance(num, np.int64)
    seq = [ord(x)-ord('0')+2 for x in str(num)]
    seq.append(EOS)
    return seq

def seq_to_int(seq):
    assert isinstance(seq, list)

    eos = 0
    while eos < len(seq) and seq[eos] not in [0,1]:
        eos += 1

    return sum([int(10**d * (x-2)) for d, x in enumerate(seq[:eos][::-1])])


def next_seqs(batch_size=100, seq_len=8, max_digits=10):
    input_seqs = []
    outputs_seqs = []
    seq_strs = []
    for b in range(batch_size):
        ints = []
        seq_str = ''

        while True:
            ints, seq_str = generate_seq(seq_len+1)
            if max(ints) < 1000 and min(ints) > 0:
                break

        seq = [int_to_seq(x) for x in ints]

        flat_inp = sum([s for s in seq[:-1]], []) + [PAD]
        flat_out = sum([s for s in seq[1:]], []) + [PAD] * 2

        input_seqs.append(flat_inp)
        outputs_seqs.append(output_seq)
        seq_strs.append(seq_str)

    max_inp_len = max([len(x) for x in input_seqs])
    max_out_len = max([len(x) for x in outputs_seqs])

    batch_input = np.zeros(shape=[batch_size, map_inp_len])
    batch_output = np.zeros(shape=[batch_size, max_out_len])

    for i in range(len(input_seqs)):
        for j in range(len(input_seqs[i])):
            batch_input[i][j] = input_seqs[i][j]
    for i in range(len(outputs_seqs)):
        for j in range(len(outputs_seqs[i])):
            batch_output[i][j] = outputs_seqs[i][j]
    return batch_input, batch_output, seq_strs


ian = IntegerAutoEncoder()

saver = tf.train.Saver(tf.global_variables())

for i in range(50001):
    xbatch, ybatch, seqs = next_seqs()

    if i % 100 == 0:
        pred = ian.predict(xbatch[0])
        print('Input', [seq_to_int(list(s)) for s in xbatch[0]])
        print('Output', [seq_to_int(list(s)) for s in ybatch[0]])
        print('Pred', [seq_to_int(list(s)) for s in pred.T])
        print('Pat', seqs[0])

    loss, summary = seq_model.train_batch(xbatch, ybatch)
        
    if i % 100 == 0:
        summary_writer.add_summary(summary, i)
        print(i, loss)
        saver.save(sess, "saved/model.ckpt")




