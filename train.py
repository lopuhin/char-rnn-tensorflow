import tensorflow as tf

import argparse
import time
import os
import cPickle

import numpy as np

from utils import TextLoader
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'w') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for e in xrange(args.num_epochs):
            sess.run(tf.assign(
                model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.prev_state.eval()
            for b in xrange(data_loader.num_batches):
                start = time.time()
                xs, ys = data_loader.next_batch()
                states = [state]
                # forward pass
                for x in xs:
                    feed = {
                        model.input_data: x,
                        model.prev_state: state}
                    state = sess.run(model.next_state, feed)
                    states.append(state)
                states, _final_state = states[:-1], states[-1]
                # backward pass
                losses = []
                prev_grads = [v.eval() for v in model.prev_grads]
                for state, x, y in reversed(zip(states, xs, ys)):
                    feed = {
                        model.input_data: x,
                        model.prev_state: state,
                        model.target: y}
                    feed.update(zip(model.prev_grads, prev_grads))
                    # FIXME - do we really apply previous char losses?
                    result = sess.run(
                        [model.train_op, model.cost] + model.grads, feed)
                    loss, prev_grads = result[1], result[2:]
                    losses.append(loss)
                train_loss = np.mean(losses)
                end = time.time()
                print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start)
                if (e * data_loader.num_batches + b) % args.save_every == 0:
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    print "model saved to {}".format(checkpoint_path)


if __name__ == '__main__':
    main()
