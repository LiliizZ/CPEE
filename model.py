from mi import Mi_estimator
from mi1 import Mi_estimator1
from module import *
import numpy as np
from module import cause_attention, cause_predict




class Article(object):
    def __init__(self, args):
        self.args = args
        self.x = tf.compat.v1.placeholder(tf.int32, [None, None], name="x_fact")  # fact
        
        self.x_plai = tf.compat.v1.placeholder(tf.int32, [None, None], name="x_plai")  # plaintiff
        self.x_defe = tf.compat.v1.placeholder(tf.int32, [None, None], name="x_defe")  # defendant
        self.x_plea = tf.compat.v1.placeholder(tf.int32, [None, None], name="x_plea")  # plea
        self.cause_class = tf.compat.v1.placeholder(tf.int32, [None, None, None], name="cause_emb")
        self.cause4_class = tf.compat.v1.placeholder(tf.int32, [None, args.num_cause4, None], name="cause4_emb")
        self.cause3_class = tf.compat.v1.placeholder(tf.int32, [None, args.num_cause3, None], name="cause3_emb")
        self.x_gen_class = tf.compat.v1.placeholder(tf.int32, [None, args.num_outputs2, None], name="articles_gen_emb")
        self.x_class = tf.compat.v1.placeholder(tf.int32, [None, args.num_outputs1, None], name="articles_spe_emb")

        self.y1_ = tf.compat.v1.placeholder(tf.float32, [None, args.num_outputs1], name="y1")  # sepcific
        self.y2_ = tf.compat.v1.placeholder(tf.float32, [None, args.num_outputs2], name="y2")  # general
        self.y_cause4 = tf.compat.v1.placeholder(tf.int32, [None, args.num_cause4], name="y_cause4")
        self.y_cause3 = tf.compat.v1.placeholder(tf.int32, [None, args.num_cause3], name="y_cause3")
        self.y_cause = tf.compat.v1.placeholder(tf.int32, [None, args.num_cause], name="y_cause")
        self.y_jg = tf.compat.v1.placeholder(tf.int32, [None, args.num_judg], name="judg")
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.is_training = tf.compat.v1.placeholder(tf.bool, name="is_training")
        self.task_weight = tf.Variable(1.0, trainable=True, name="task_weight")
        #self.l2_reg_strength = tf.Variable(1.0, trainable=False, name="l2_reg_strength")
        self.pretrained_embedding = np.load("/data/liliz/hecp/embedding_matrix.npy", allow_pickle=True) # word2vec embeddings
        
        self._build_graph(is_training=True)

    def _build_graph(self, is_training):

        with tf.name_scope("embed"):
            if self.pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([self.args.vocab_size, self.args.embed_word_dim], minval=-1.0, maxval=1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if self.args.embedding_type == 0:
                    self.embedding = tf.constant(self.pretrained_embedding, dtype=tf.float32, name="embedding")
                if self.args.embedding_type == 1:
                    self.embedding = tf.Variable(self.pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
                    
            embedding_inputs_fact = tf.nn.embedding_lookup(self.embedding, self.x)
            embedding_inputs_plai = tf.nn.embedding_lookup(self.embedding, self.x_plai)
            embedding_inputs_defe = tf.nn.embedding_lookup(self.embedding, self.x_defe)
            embedding_inputs_plea = tf.nn.embedding_lookup(self.embedding, self.x_plea)
            
            self.embedding_article = tf.compat.v1.get_variable('embedding_article', [self.args.article_vocab_size + 1, self.args.embed_word_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-3), trainable=True)

            self.embedding_cause = tf.compat.v1.get_variable('embedding_cause', [self.args.cause_vocab_size + 1, self.args.embed_word_cause], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=1e-3), trainable=True)#1e-4

            x_class = tf.reduce_mean(self.x_class, -1)
            W_class = tf.nn.embedding_lookup(self.embedding_article, x_class)
            x_gen_class = tf.reduce_mean(self.x_gen_class, -1)
            W_gen_class = tf.nn.embedding_lookup(self.embedding_article, x_gen_class)
            cause_class = tf.reduce_mean(self.cause_class, -1)
            W_cause_class = tf.nn.embedding_lookup(self.embedding_cause, cause_class)  
            cause4_class = tf.reduce_mean(self.cause4_class, -1)
            W_cause4_class = tf.nn.embedding_lookup(self.embedding_cause, cause4_class)
            cause3_class = tf.reduce_mean(self.cause3_class, -1)
            W_cause3_class = tf.nn.embedding_lookup(self.embedding_cause, cause3_class)
            print("article_embedding", W_class.shape, W_gen_class.shape) #article_embedding
            print("cause_embedding", W_cause4_class.shape, W_cause3_class.shape) #cause_embedding
            '''if is_training and self.config.keep_prob < 1:
                embedding_inputs_fact = tf.nn.dropout(embedding_inputs_fact, self.config.keep_prob)
                embedding_inputs_plai = tf.nn.dropout(embedding_inputs_plai, self.config.keep_prob)
                embedding_inputs_defe = tf.nn.dropout(embedding_inputs_defe, self.config.keep_prob)
                embedding_inputs_plea = tf.nn.dropout(embedding_inputs_plea, self.config.keep_prob)

                W_class = tf.nn.dropout(W_class, self.config.keep_prob)
                W_gen_class = tf.nn.dropout(W_gen_class, self.config.keep_prob)
                W_cause_class = tf.nn.dropout(W_cause_class, self.config.keep_prob)
                W_cause4_class = tf.nn.dropout(W_cause4_class, self.config.keep_prob)
                W_cause3_class = tf.nn.dropout(W_cause3_class, self.config.keep_prob)'''


        # Bi-LSTM Layer
        with tf.name_scope("Bi-lstm"):
            ###fact bi-lstm
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.args.lstm_hidden_size)  # forward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(self.args.lstm_hidden_size)  # backward direction cell
            if self.keep_prob is not None:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,
                                                             output_keep_prob=self.keep_prob)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,
                                                             output_keep_prob=self.keep_prob)

            # Creates a dynamic bidirectional recurrent neural network
            # shape of `outputs`: tuple -> (outputs_fw, outputs_bw)
            # shape of `outputs_fw`: [batch_size, sequence_length, lstm_hidden_size]

            # shape of `state`: tuple -> (outputs_state_fw, output_state_bw)
            # shape of `outputs_state_fw`: tuple -> (c, h) c: memory cell; h: hidden state
            
            output_fact, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                             embedding_inputs_fact, dtype=tf.float32)
            output_plai, state_plai = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                             embedding_inputs_plai, dtype=tf.float32)
            output_defe, state_defe = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                            embedding_inputs_defe, dtype=tf.float32)
            output_plea, state_plea = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                              embedding_inputs_plea, dtype=tf.float32)
            # Concat output
            self.lstm_out_fact = tf.concat(output_fact, axis=2, name="concat-fact")  # [batch_size, sequence_length, lstm_hidden_size * 2]
            self.lstm_out_pool = tf.reduce_mean(self.lstm_out_fact, axis=1)  # [batch_size, lstm_hidden_size * 2]

            self.lstm_out_plai = tf.concat(output_plai, axis=2, name="concat-plai")
            self.lstm_out_plai_pool = tf.reduce_mean(self.lstm_out_plai, axis=1)

            self.lstm_out_defe = tf.concat(output_defe, axis=2, name="concat-defe")
            self.lstm_out_defe_pool = tf.reduce_mean(self.lstm_out_defe, axis=1)

            #self.lstm_out_plea = tf.concat(output_plea, axis=2)#??256
            self.lstm_out_plea =  tf.concat(output_plea, axis=2, name="concat-plea")
            self.lstm_out_plea_pool = tf.reduce_mean(self.lstm_out_plea, axis=1)  # ?256
            

        with tf.name_scope("hierarchical-cause"):

            #self.share_input = tf.layers.dense(self.lstm_out_fact, self.args.fc_hidden_size, activation=tf.nn.relu, name="share_articles") 
            ###input for cause
            self.lstm_cause = tf.concat([self.lstm_out_fact, self.lstm_out_plai, self.lstm_out_defe], axis=1) # self.lstm_out_plea
   
            # fourth Level
            self.fourth_att_weight, self.fourth_att_out = cause_attention(self.lstm_cause, W_cause4_class,
                                                                     self.args.attention_unit_size, name="fourth-")
            # ?,?,c,s ; ?,c,e
            self.fourth_local_input = tf.concat([self.lstm_cause, self.fourth_att_out], axis=1)
            # ?,s+c,e
            self.fourth_local_fc_out = tf.layers.dense(self.fourth_local_input, self.args.fc_hidden_size, activation=tf.nn.relu, name="fourth-local-fc") #_fc_layer(self.fourth_local_input, self.config, name="fourth-local-")
            # ?,s+c,e
            self.fourth_local_fc_out = tf.nn.dropout(self.fourth_local_fc_out, self.keep_prob)

            self.fourth_logits, self.fourth_scores, self.fourth_visual = cause_predict(
                self.fourth_local_fc_out, self.fourth_att_weight, self.args.num_cause4, self.keep_prob,
                name="fourth-")
            #(?, 5) (?, 5, ?)
            self.pred_cause4 = tf.argmax(self.fourth_scores, 1, name="cause4_output")
            self.y_cause4 = tf.cast(self.y_cause4, tf.float32)
            self.cause4_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y_cause4, 1), self.pred_cause4), tf.float32))

            # third Level
            #self.third_att_input = tf.matmul(self.fourth_visual, self.share_input)#tf.expand_dims(self.fourth_visual, -1)
            #(?, 5, 512)
            self.third_att_weight, self.third_att_out = cause_attention(self.lstm_cause, W_cause3_class, 
                                                                        self.args.attention_unit_size, name="third--")
            
            self.third_local_input = tf.concat([self.lstm_cause, self.third_att_out], axis=1)
            self.third_local_fc_out = tf.layers.dense(self.third_local_input, self.args.fc_hidden_size, activation=tf.nn.relu, name="third-local-fc") #_fc_layer(self.third_local_input, self.config, name="third-local-")

            self.third_local_fc_out = tf.nn.dropout(self.third_local_fc_out , self.keep_prob)
            
            self.third_logits, self.third_scores, self.third_visual = cause_predict(
                self.third_local_fc_out, self.third_att_weight, self.args.num_cause3, self.keep_prob,
                name="third-")
            self.pred_cause3 = tf.argmax(self.third_scores, 1, name="cause3_output")
            self.y_cause3 = tf.cast(self.y_cause3, tf.float32)
            self.cause3_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y_cause3, 1), self.pred_cause3), tf.float32))

          
            '''print("fourth_out, third_out", self.fourth_local_fc_out.shape,
                  self.third_local_fc_out.shape)'''
            self.ham_out = tf.concat([self.fourth_local_fc_out, self.third_local_fc_out
                                      ], axis=1)
            # Fully Connected Layer
            self.fc_out = tf.layers.dense(self.ham_out, self.args.fc_hidden_size, activation=tf.nn.relu, name="local-all-fc")#_fc_layer(self.ham_out, self.config, name="global-")
            
   

        # Add dropout
        with tf.name_scope("dropout1"):
            self.h_drop = tf.nn.dropout(self.fc_out, self.keep_prob)
            print("----h_drop", self.h_drop.shape)

        # Global scores
        with tf.name_scope("global-cause"):
            #self.global_input = tf.reduce_mean(self.h_drop, axis=1)
            self.global_att_weight, self.global_att_out = cause_attention(self.lstm_cause, W_cause_class,
                                                                     self.args.attention_unit_size, name="global-cause-")
            
            self.global_input = tf.concat([self.lstm_cause, self.global_att_out], axis=1)
            # ?,s+c,e 
            self.global_fc_out = tf.layers.dense(self.global_input, self.args.fc_hidden_size, activation=tf.nn.relu, name="global-cause-fc") # ?,s+c,e
         
            self.cause_global_logits, self.cause_global_scores, self.global_visual = cause_predict(
                self.global_fc_out, self.global_att_weight, self.args.num_cause, self.keep_prob, name="global-cause-")
        
            self.pred_cause = tf.argmax(self.cause_global_scores, 1, name="cause_output")
            self.y_cause = tf.cast(self.y_cause, tf.float32)
            self.cause_acc = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.y_cause, 1), self.pred_cause), tf.float32))


        with tf.name_scope("articles"):
            #input for articles
            self.share_input = tf.layers.dense(self.lstm_out_fact, self.args.fc_hidden_size, activation=tf.nn.relu, name="share_articles") 
            
            G_s, output1 = general_encoder(self.share_input, W_class, self.args.num_outputs1, name="specific")
            print("output_s:", output1.shape)
            self.a1_scores = tf.sigmoid(output1, name="scores1")
           
            
            G_g, output2 = general_encoder(self.share_input, W_gen_class, self.args.num_outputs2, name="general")
            self.a2_scores = tf.sigmoid(output2, name="scores2")
            print("output_g:", output2.shape)

        with tf.name_scope("gate"):
            G_s_tran = tf.transpose(G_s, [0, 2, 1])  # e * c
            c_s = tf.contrib.keras.backend.dot(G_s_tran, self.lstm_out_fact)
            c_s = tf.reduce_mean(c_s, axis=2)#76,256
            c_s1 = tf.transpose(c_s, [0, 2, 1])#256,76
            s_W2 = tf.layers.dense(c_s1, 1, use_bias=False, name="Specific_W2")#256,1
            S_W2 = tf.transpose(s_W2, [0, 2, 1])#1*256
            print("c_s, W2", c_s.shape, s_W2.shape)  # 76,256 #256,200

            G_g_tran = tf.transpose(G_g, [0, 2, 1])  # e * c
            c_g = tf.contrib.keras.backend.dot(G_g_tran, self.lstm_out_fact)
            c_g = tf.reduce_mean(c_g, axis=2)#129*256
            c_g1 = tf.transpose(c_g, [0, 2, 1]) #256 129
            g_W3 = tf.layers.dense(c_g1, 1, use_bias=False, name="General_W3")  #256 1
            G_W3 = tf.transpose(g_W3, [0, 2, 1])  # 1*256
            gate = tf.nn.sigmoid(S_W2 + G_W3)
            c_a = tf.concat([gate * S_W2 , (1 - gate) * G_W3], axis=1)#1
            #c_a = tf.layers.dense(c_a, 2*self.config.lstm_hidden_size, activation=tf.nn.relu, name="gate_dense")
            

        with tf.name_scope("judgment"):
            ###input for final judgment
            fact_plea = coattention(self.lstm_out_plea, self.lstm_out_fact)#coattention(self.lstm_out_plea, self.lstm_out_fact)
            fact_plai = coattention(self.lstm_out_fact, self.lstm_out_plai)
            fact_defe = coattention(self.lstm_out_fact, self.lstm_out_defe)
            fact_article = coattention(self.lstm_out_fact, c_a)



        with tf.name_scope("predit_plea"):
            judg_final_state = tf.concat([fact_plea, fact_plai, fact_defe, fact_article], axis=1)
            
            judg_medi_state = tf.layers.dense(judg_final_state, self.args.fc_hidden_size, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001), name="medi")
            judg_final_state_out = tf.nn.dropout(judg_medi_state, self.keep_prob)
            judg_final = tf.reduce_mean(judg_final_state_out, axis=1)
            self.plea_logits = tf.layers.dense(judg_final, self.args.num_judg, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001), name="judg")
            self.pred_jg = tf.nn.softmax(self.plea_logits, name="predict_prob")
            self.pred_plea = tf.argmax(self.pred_jg, axis=1, name="plea_predict") # 预测类别



        with tf.name_scope("mi"):

            my_mi_estimator = Mi_estimator(dim_x=[self.args.maxlen, self.args.embed_word_dim],
                                           dim_z=[self.args.maxlen, self.args.embed_word_dim],
                                           batch_size=self.args.batch_size)
            my_mi_estimator1 = Mi_estimator1(dim_x=[self.args.maxlen, self.args.embed_word_dim],
                                           dim_z=[self.args.maxlen, self.args.embed_word_dim],
                                           batch_size=self.args.batch_size)
            lstm_out_fact = tf.cast(self.lstm_out_fact, dtype=tf.float32)
            lstm_out_plai = tf.cast(self.lstm_out_plai, dtype=tf.float32)
            lstm_out_defe = tf.cast(self.lstm_out_defe, dtype=tf.float32)
            self.x1 = tf.cast(embedding_inputs_fact, dtype=tf.float32)
            self.x_plai1 = tf.cast(embedding_inputs_plai, dtype=tf.float32)
            self.x_defe1 = tf.cast(embedding_inputs_defe, dtype=tf.float32)

            self.mi_pf_op, \
            self.mi_pf_quantities = my_mi_estimator(lstm_out_fact, lstm_out_plai)
            self.mi_df_op, \
            self.mi_df_quantities = my_mi_estimator1(lstm_out_fact, lstm_out_defe)

        with tf.name_scope("all-loss"):
            def cal_loss(labels, logits, name):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                losses = tf.reduce_mean(losses, name=name + "losses")
                return losses
            
            self.plea_loss = cal_loss(self.y_jg, self.plea_logits, name="jg_")

            #self.plea_loss = seq2seq.sequence_loss(logits=self.plea1_logits, targets=self.y_jg1, weights=self.plea_sample_mask)
            self.mi_pf_loss = self.mi_pf_quantities['mi_for_grads']
            self.mi_df_loss = self.mi_df_quantities['mi_for_grads']
            self.regu_loss = self.mi_pf_loss + self.mi_df_loss

            self.jg_loss = tf.add_n([self.plea_loss, 0.001 * self.regu_loss], name="jg_all_losses")
            
            self.c4_loss = cal_loss(labels=self.y_cause4, logits=self.fourth_logits, name="fourth_")
            self.c3_loss = cal_loss(labels=self.y_cause3, logits=self.third_logits, name="third_")
            self.cause_local_loss = tf.add_n([self.c4_loss, self.c3_loss], name="local_losses")
            # Global Loss
            self.cause_global_loss = cal_loss(labels=self.y_cause, logits=self.cause_global_logits, name="global_loss")
            self.cause_loss = tf.add_n([self.cause_global_loss, 0.1 * self.cause_local_loss], name="cause_loss")
            self.a1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y1_, logits=output1))
            self.a2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y2_, logits=output2))
            self.article_loss = tf.add_n([self.a1_loss, self.a2_loss], name="article_loss")
            self.joint_loss = tf.add_n([self.jg_loss, self.cause_loss, 0.5 * self.article_loss], name="loss")#, self.cause_loss, self.a1_loss, self.a2_loss
            
