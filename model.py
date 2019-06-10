import tensorflow as tf
from tensorflow.contrib import rnn
from data_processing import get_init_embedding


class Model(object):
    def __init__(self, index2word, question_max_len, answer_max_len, args, forward_only=False):
        self.vocabulary_size = len(index2word)
        self.embedding_size = args.embedding_size
        self.num_hidden = args.num_hidden
        self.num_layers = args.num_layers
        self.learning_rate = args.learning_rate
        self.beam_width = 1
        if not forward_only:
            self.keep_prob = args.keep_prob
        else:
            self.keep_prob = 1.0
        self.cell = tf.nn.rnn_cell.LSTMCell
        with tf.variable_scope("decoder/projection"):
            self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

        self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
        self.X = tf.placeholder(tf.int32, [None, question_max_len])
        self.X_len = tf.placeholder(tf.int32, [None])
        self.decoder_input = tf.placeholder(tf.int32, [None, answer_max_len])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_target = tf.placeholder(tf.int32, [None, answer_max_len])
        self.global_step = tf.Variable(0, trainable=False)

        with tf.name_scope("embedding"):
            init_embeddings = tf.constant(get_init_embedding(index2word, self.embedding_size), dtype=tf.float32)

            self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
            self.encoder_emb_input = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X), perm=[1, 0, 2])
            self.decoder_emb_input = tf.transpose(tf.nn.embedding_lookup(
                self.embeddings, self.decoder_input), perm=[1, 0, 2])

        with tf.name_scope("encoder"):
            fw_cells = [rnn.DropoutWrapper(self.cell(self.num_hidden)) for _ in range(self.num_layers)]
            bw_cells = [rnn.DropoutWrapper(self.cell(self.num_hidden)) for _ in range(self.num_layers)]

            encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cells, bw_cells, self.encoder_emb_input,
                sequence_length=self.X_len, time_major=True, dtype=tf.float32)
            self.encoder_output = tf.concat(encoder_outputs, 2)
            encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
            encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
            self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

        with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
            decoder_cell = self.cell(self.num_hidden * 2)

            if not forward_only:
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
                attention_mechanism = self.get_attention_mechanism(attention_states, self.X_len, normalize=True)
                decoder_cell = self.add_attention(decoder_cell, attention_mechanism)
                initial_state = self.get_initial_state(decoder_cell, self.batch_size, self.encoder_state)
                helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_input, self.decoder_len, time_major=True)
                decoder = self.get_decoder(decoder_cell, initial_state, with_beam_search=False, helper=helper)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)
                self.decoder_output = outputs.rnn_output
                self.logits = tf.transpose(
                    self.projection_layer(self.decoder_output), perm=[1, 0, 2])
                self.logits_reshape = tf.concat([self.logits, tf.zeros(
                    [self.batch_size, answer_max_len - tf.shape(self.logits)[1], self.vocabulary_size])], axis=1)
            else:
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
                tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
                    self.encoder_state, multiplier=self.beam_width)
                tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)
                attention_mechanism = self.get_attention_mechanism(tiled_encoder_output, tiled_seq_len, normalize=True)
                decoder_cell = self.add_attention(decoder_cell, attention_mechanism)
                initial_state = self.get_initial_state(
                    decoder_cell, self.batch_size * self.beam_width, tiled_encoder_final_state)
                decoder = self.get_decoder(decoder_cell, initial_state, with_beam_search=True)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder, output_time_major=True, maximum_iterations=answer_max_len, scope=decoder_scope)
                self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

        with tf.name_scope("loss"):
            if not forward_only:
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_reshape, labels=self.decoder_target)
                weights = tf.sequence_mask(self.decoder_len, answer_max_len, dtype=tf.float32)
                self.loss = tf.reduce_sum(cross_entropy * weights / tf.to_float(self.batch_size))

                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def add_attention(self, decoder_cell, attention_mechanism):
        return tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism, attention_layer_size=self.num_hidden * 2)

    def get_initial_state(self, decoder_cell, batch_size, cell_state):
        return decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size).clone(cell_state=cell_state)

    def get_attention_mechanism(self, memory, memory_sequence_len, normalize):
        return tf.contrib.seq2seq.BahdanauAttention(
            self.num_hidden * 2, memory, memory_sequence_length=memory_sequence_len, normalize=normalize)

    def get_decoder(self, decoder_cell, initial_state, with_beam_search=False, helper=None):
        if with_beam_search:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=self.embeddings,
                start_tokens=tf.fill([self.batch_size], tf.constant(2)),
                end_token=tf.constant(3),
                initial_state=initial_state,
                beam_width=self.beam_width,
                output_layer=self.projection_layer
            )
        else:
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
        return decoder
