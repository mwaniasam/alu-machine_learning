#!/usr/bin/env python3
"""
Module containing the full Transformer network architecture.
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates Bahdanau attention for a sequence model."""
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(
            units, kernel_initializer='glorot_uniform'
        )
        self.U = tf.keras.layers.Dense(
            units, kernel_initializer='glorot_uniform'
        )
        self.V = tf.keras.layers.Dense(
            1, kernel_initializer='glorot_uniform'
        )

    def call(self, s_prev, hidden_states):
        """Calculates attention context and alignment weights."""
        s_prev_time = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(s_prev_time) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi-head attention."""
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(
            dm, kernel_initializer='glorot_uniform'
        )
        self.Wk = tf.keras.layers.Dense(
            dm, kernel_initializer='glorot_uniform'
        )
        self.Wv = tf.keras.layers.Dense(
            dm, kernel_initializer='glorot_uniform'
        )
        self.linear = tf.keras.layers.Dense(
            dm, kernel_initializer='glorot_uniform'
        )

    def split_heads(self, x, batch_size):
        """Splits the last dimension into (h, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Executes multi-head attention processing logic."""
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        scaled_attention = tf.matmul(weights, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dm))
        return self.linear(concat_attention), weights


class EncoderBlock(tf.keras.layers.Layer):
    """An encoder block for a transformer network."""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu', kernel_initializer='glorot_uniform'
        )
        self.dense_output = tf.keras.layers.Dense(
            dm, kernel_initializer='glorot_uniform'
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Processes input through MHA and Feed-Forward sublayers."""
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class DecoderBlock(tf.keras.layers.Layer):
    """A decoder block for a transformer network."""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu', kernel_initializer='glorot_uniform'
        )
        self.dense_output = tf.keras.layers.Dense(
            dm, kernel_initializer='glorot_uniform'
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Processes input through masked MHA, cross MHA, and FFN sublayers."""
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)


def positional_encoding(max_seq_len, dm):
    """Calculates fixed sinusoidal positional encodings."""
    PE = tf.zeros((max_seq_len, dm), dtype=tf.float32)
    position = tf.cast(tf.range(max_seq_len)[:, tf.newaxis], tf.float32)
    div_term = tf.exp(
        tf.cast(tf.range(0, dm, 2), tf.float32) * -(tf.math.log(10000.0) / dm)
    )
    PE_pos = tf.cast(position * div_term, tf.float32)
    PE_even = tf.sin(PE_pos)
    PE_odd = tf.cos(PE_pos)
    PE = tf.reshape(
        tf.concat([PE_even[:, :, tf.newaxis], PE_odd[:, :, tf.newaxis]], axis=2),
        (max_seq_len, dm)
    )
    return PE


class Encoder(tf.keras.layers.Layer):
    """A full Transformer Encoder containing N blocks."""
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Processes input indices through encoder stack."""
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """A full Transformer Decoder containing N blocks."""
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Processes target indices through decoder stack."""
        target_seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:target_seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](
                x, encoder_output, training, look_ahead_mask, padding_mask
            )
        return x


class Transformer(tf.keras.Model):
    """A full Transformer Network combining Encoder and Decoder."""
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate
        )
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate
        )
        self.linear = tf.keras.layers.Dense(
            target_vocab, kernel_initializer='glorot_uniform'
        )

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """Executes a full forward pass through the Transformer."""
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(
            target, enc_output, training, look_ahead_mask, decoder_mask
        )
        return self.linear(dec_output)
