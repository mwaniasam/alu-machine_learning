#!/usr/bin/env python3
"""
Module to create and train a Transformer model for machine translation.
"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom learning rate scheduler based on Attention Is All You Need.
    """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss_object):
    """
    Calculates sparse categorical crossentropy loss ignoring padded tokens.
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a Transformer model for Portuguese to English translation.
    """
    # 1. Instantiate the dataset pipeline
    data = Dataset(batch_size, max_len)
    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    # 2. Instantiate the full Transformer model
    transformer = Transformer(
        N=N, dm=dm, h=h, hidden=hidden,
        input_vocab=input_vocab, target_vocab=target_vocab,
        max_seq_input=max_len, max_seq_target=max_len
    )

    # 3. Configure Optimizers and Loss Tracking Objects
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    # Metrics trackers
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy'
    )

    # 4. Set Up Vectorized Custom Step Execution
    @tf.function
    def train_step(inp, tar):
        # Teacher forcing: shifted inputs pass into the model
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions = transformer(
                inp, tar_inp, True, enc_mask, combined_mask, dec_mask
            )
            loss = loss_function(tar_real, predictions, loss_object)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(tar_real, predictions)

    # 5. Iterative Epoch Execution Loop
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch, (inp, tar) in enumerate(data.data_train):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(
                    "Epoch {}, batch {}: loss {:.8f} accuracy {:.8f}".format(
                        epoch + 1, batch,
                        train_loss.result(), train_accuracy.result()
                    )
                )

        print(
            "Epoch {}: loss {:.8f} accuracy {:.8f}".format(
                epoch + 1, train_loss.result(), train_accuracy.result()
            )
        )

    return transformer
