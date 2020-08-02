# Author : Samantha Mahendran for RelEx-BERT
import tensorflow as tf
from BERT.bert_layer import BertLayer

class Run_BERT:
    def __init__(self, bert_model):
        # Initialize session
        sess = tf.Session()
        self.bert_model = bert_model

        model = self. build_model(max_seq_length)
        # Instantiate variables
        self.initialize_vars(sess)

        model.fit(
            [self.bert_model.train_input_ids, self.bert_model.train_input_masks, self.bert_model.train_segment_ids],
            self.bert_model.train_labels,
            validation_data=([self.bert_model.test_input_ids, self.bert_model.test_input_masks, self.bert_model.test_segment_ids], self.bert_model.test_labels),
            epochs=1,
            batch_size=32
        )

    # Build model
    def build_model(self, max_seq_length):
        in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        bert_output = BertLayer(n_fine_tune_layers=3, pooling="first",bert_path=self.bert_model.bert_path)(bert_inputs)
        dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
        pred = tf.keras.layers.Dense(1, activation='softmax')(dense)

        model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model


    def initialize_vars(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        K.set_session(sess)

