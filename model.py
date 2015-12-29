import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import numpy as np


class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size])
        self.target = tf.placeholder(tf.int32, [args.batch_size])
        self.prev_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable(
                'softmax_w', [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable(
                'softmax_b', [args.vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    'embedding', [args.vocab_size, args.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)
        # TODO - use loop

        self.output, self.next_state = cell(inputs, self.prev_state)
        self.logits = tf.nn.xw_plus_b(self.output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        # TODO - use tf.nn.softmax_cross_entropy_with_logits
        loss = seq2seq.sequence_loss_by_example(
            [self.logits],
            [self.target],
            [tf.ones([args.batch_size])],
            args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # FIXME???
        # self.prev_grads = [tf.ones([], dtype=tf.float32)
        self.grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), #self.prev_grads),
            args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(self.grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The '):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
            # sample = int(np.random.choice(len(p), p=p))
            sample = weighted_pick(p)
            pred = chars[sample]
            ret += pred
            char = pred
        return ret
