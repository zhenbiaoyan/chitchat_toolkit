import time
start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
from model import Model
from util import build_word_dict, build_dataset, batch_iter, load_qa_list, get_feed_dict
import time

# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
# tf.logging.set_verbosity(tf.logging.FATAL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=3, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=10, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
with open("args.pickle", "wb") as f:
    pickle.dump(args, f)

if not os.path.exists("saved_model"):
    os.mkdir("saved_model")

print("Building dictionary...")
word2index, index2word = build_word_dict("train", "data/train")

question_max_len, answer_max_len = 50, 50
print("Loading training dataset...")
qa_list = load_qa_list('data/train')
train_x, train_y = build_dataset("train", qa_list, word2index, question_max_len, answer_max_len)


with tf.Session() as sess:
    start = time.time()
    model = Model(index2word, question_max_len, answer_max_len, args)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
    num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

    print("\nIteration starts.")
    print("Number of batches per epoch :", num_batches_per_epoch)
    for batch_x, batch_y in batches:
        train_feed_dict = get_feed_dict(model, word2index, answer_max_len, batch_x, batch_y)

        _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

        if step % num_batches_per_epoch == 0:
            print("step {0}: loss = {1}".format(step, loss))
            hours, rem = divmod(time.perf_counter() - start, 3600)
            minutes, seconds = divmod(rem, 60)
            saver.save(sess, "./saved_model/model.ckpt", global_step=step)
            print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
                  "Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")

    print("Training time: {0}".format(str(time.time() - start)))
