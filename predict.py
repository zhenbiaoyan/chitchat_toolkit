import tensorflow as tf
import pickle
from model import Model
from util import build_word_dict, build_dataset, batch_iter, load_qa_list, get_feed_dict


with open("args.pickle", "rb") as f:
    args = pickle.load(f)

word2index, index2word = build_word_dict("dev", "data/dev")
question_max_len, answer_max_len = 50, 50
qa_list = load_qa_list('data/dev')
dev_x, dev_y = build_dataset("dev", qa_list, word2index, question_max_len, answer_max_len)


with tf.Session() as sess:
    print("Loading saved model...")
    model = Model(index2word, question_max_len, answer_max_len, args, forward_only=True)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("./saved_model/")
    saver.restore(sess, ckpt.model_checkpoint_path)

    batches = batch_iter(dev_x, dev_y, args.batch_size, 1)

    print("Writing Answers to 'result.txt'...")
    for batch_x, batch_y in batches:
        batch_x_len = list(map(lambda x: len([xx for xx in x if xx != 0]), batch_x))

        dev_feed_dict = {
            model.batch_size: len(batch_x),
            model.X: batch_x,
            model.X_len: batch_x_len,
        }
        prediction = sess.run(model.prediction, feed_dict=dev_feed_dict)
        prediction_output = [[index2word[index] for index in ans] for ans in prediction[:, 0, :]]

        with open("result_beam_1.txt", "a") as f:
            for i, line in enumerate(prediction_output):
                answer = list()
                for word in line:
                    if word == "</s>":
                        break
                        print("</s>")
                    answer.append(word)
                print(" ".join(answer), file=f)

    print('Answers are saved to "result.txt"...')