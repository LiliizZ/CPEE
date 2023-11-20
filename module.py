import tensorflow as tf
from utils.ops import get_shape_list

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


####articles
def general_encoder(encoding, W_class, num_output, name=""):
    with tf.variable_scope("encoder" + name):
        W_class = tf.cast(W_class, tf.float32)
        W_class_tran = tf.transpose(W_class, [0, 2, 1])  # e * c
        G = tf.matmul(encoding, W_class_tran) 
        G_ = tf.reduce_mean(G, axis=1)
        G_output = tf.layers.dense(G_, num_output)
        return G, G_output

def cause_attention(input_x, w_cause, attention_unit_size, name=""):
    """
    Attention Layer.

    Args:
        input_x: [batch_size, sequence_length, lstm_hidden_size * 2]
        num_classes: The number of i th level classes.
        name: Scope name.
    Returns:
        attention_weight: [batch_size, num_classes, sequence_length]
        attention_out: [batch_size, lstm_hidden_size * 2]
    """
    num_units = input_x.get_shape().as_list()[-1]
    with tf.name_scope(name + "attention"):
        W_s1 = tf.Variable(tf.truncated_normal(shape=[attention_unit_size, num_units],
                                               stddev=0.1, dtype=tf.float32), name="W_s1")
        '''W_s2 = tf.Variable(tf.truncated_normal(shape=[num_classes, opt.attention_unit_size],
                                               stddev=0.1, dtype=tf.float32), name="W_s2")'''
        # attention_matrix: [batch_size, num_classes, sequence_length]
        attention_matrix = tf.map_fn(
            fn=lambda x: tf.matmul(w_cause, x),
            elems=tf.tanh(
                tf.map_fn(
                    fn=lambda x: tf.matmul(W_s1, tf.transpose(x)),
                    elems=input_x,
                    dtype=tf.float32
                )
            )
        )
        attention_weight = tf.nn.softmax(attention_matrix, name="attention")
        attention_out = tf.matmul(attention_weight, input_x)
        attention_out = tf.reduce_mean(attention_out, axis=1)
    return attention_weight, attention_out



def cause_predict(input_x, input_att_weight, num_classes, keep_prob, name=""):
    """
    Local Layer.

    Args:
        input_x: [batch_size, fc_hidden_size]
        input_att_weight: [batch_size, num_classes, sequence_length]
        num_classes: Number of classes.
        name: Scope name.
    Returns:
        logits: [batch_size, num_classes]
        scores: [batch_size, num_classes]
        visual: [batch_size, sequence_length]
    """
    with tf.name_scope(name + "output"):
        input_x = tf.reduce_mean(input_x, axis=1)
        logits = tf.layers.dense(input_x, num_classes, activation=tf.nn.relu)
        scores = tf.nn.softmax(logits)

        # shape of visual: [batch_size, sequence_length]
        visual = tf.multiply(input_att_weight, tf.expand_dims(scores, -1))
        visual = tf.nn.softmax(visual)
        visual = tf.reduce_mean(visual, axis=1)
    return logits, scores, visual


def coattention(f_encoding, p_variation):
    L = tf.matmul(f_encoding, tf.transpose(p_variation, perm=[0, 2, 1])) 
    ## shape = (batch_size, question+1, context+1)
    L_t = tf.transpose(L, perm=[0, 2, 1])
    # normalize with respect to question
    a_p = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
    # normalize with respect to context
    a_f = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
    # summaries with respect to question, (batch_size, question+1, hidden_size)
    c_p = tf.matmul(tf.transpose(f_encoding, perm=[0, 2, 1]), a_p)
    c_f_emb = tf.concat([p_variation, tf.transpose(c_p, perm=[0, 2, 1])], 2)
    # summaries of previous attention with respect to context
    a_f = tf.transpose(a_f, perm=[0, 2, 1])
    co_att = tf.matmul(a_f, c_f_emb)
    # final coattention context, (batch_size, context+1, 3*hidden_size)
    return co_att  #


def attention_mechanism(inputs, x_mask=None):
    """
    Attention mechanism layer.

    :param inputs: outputs of RNN/Bi-RNN layer (not final state)
    :param x_mask:
    :return: outputs of the passed RNN/Bi-RNN reduced with attention vector
    """
    # In case of Bi-RNN input we need to concatenate outputs of its forward and backward parts
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)
    _, sequence_length, hidden_size = get_shape_list(inputs)

    v = tf.layers.dense(
        inputs, hidden_size,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        activation=tf.tanh,
        use_bias=True
    )
    att_score = tf.layers.dense(
        v, 1,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        use_bias=False
    )  # batch_size, sequence_length, 1

    att_score = tf.squeeze(att_score, axis=-1) * x_mask + VERY_NEGATIVE_NUMBER * (
            1 - x_mask)  # [batch_size, sentence_length
    att_score = tf.expand_dims(tf.nn.softmax(att_score), axis=-1)  # [batch_size, sentence_length, 1]
    att_pool_vec = tf.matmul(tf.transpose(att_score, [0, 2, 1]), inputs)  # [batch_size,  h]
    att_pool_vec = tf.squeeze(att_pool_vec, axis=1)

    return att_pool_vec, att_score
