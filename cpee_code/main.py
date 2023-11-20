from tqdm import tqdm
from get_metrics import get_articles_result, get_cause_score, get_result

from model import *
from load_file import get_time_dif, np_read_pre_new
from utils import checkmate as cm
from utils import data_helpers as dh

import os
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#logger = dh.logger_fn("tflog", "./logs/logs.log") #.format(time.asctime())

from argments import parser
from tensorflow.python.framework import ops

ops.reset_default_graph()
g = tf.get_default_graph()
print([op.name for op in g.get_operations()])

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    return logger


def cpee():
  

    saver = tf.train.Saver()

    total_batch = 0  
    best_jg_acc = 0.0
    best_jg_pre = 0.0   
    best_jg_rec = 0.0   
    best_jg_f1 = 0.0  
    best_jg_epoch = 0

    with tf.Graph().as_default():
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        sess = tf.Session(config=config_gpu)
        with sess.as_default():
            model = Article(args)
            sess.run(init_g)
            sess.run(init_l)
            sess.run(tf.initialize_all_variables())
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=args.lr,
                                                           global_step=model.global_step,
                                                           decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate,
                                                           staircase=True)
                optim = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optim.compute_gradients(model.joint_loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=args.norm_ratio)
                train_plea_op = optim.apply_gradients(zip(grads, vars), global_step=model.global_step, name="train_plea_op1")
                #sess.run(tf.initialize_all_variables())

                learning_rate_cause = tf.train.exponential_decay(learning_rate=args.lr_cause,
                                                           global_step=model.global_step,
                                                           decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate,
                                                           staircase=True)
                optim_cause = tf.train.AdamOptimizer(learning_rate_cause)
                grads_cause, vars_cause = zip(*optim_cause.compute_gradients(model.cause_global_loss))
                grads_cause, _ = tf.clip_by_global_norm(grads_cause, clip_norm=args.norm_ratio)
                train_cause_op = optim.apply_gradients(zip(grads, vars), global_step=model.global_step, name="train_cause_op1")
                sess.run(tf.initialize_all_variables())

            
            checkpoint_prefix = os.path.join(args.checkpoint_dir, "model")
            jg_loss_summary = tf.summary.scalar("jg_loss", model.joint_loss)
            # Train summaries
            train_summary_op = tf.summary.merge([jg_loss_summary])
            train_summary_dir = os.path.join(args.path, args.summary, "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([jg_loss_summary])
            validation_summary_dir = os.path.join(args.path, args.summary, "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=args.best_checkpoint_dir, num_to_keep=3, maximize=True)


            def feed_data(x_batch, plai_batch, defe_batch, plea_batch,
                          cause_class_batch, cause4_class_batch, cause3_class_batch,
                          x_gen_class_batch, x_spe_class_batch,
                          y_cause_batch, y_cause4_batch, y_cause3_batch,
                          y1_batch, y2_batch,
                          y_jgs_batch, keep_prob):
                feed_dict = {
                    model.x: x_batch,
                    model.x_plai: plai_batch,
                    model.x_defe: defe_batch,
                    model.x_plea: plea_batch,
                    model.cause_class: cause_class_batch, #emb
                    model.cause4_class: cause4_class_batch,
                    model.cause3_class: cause3_class_batch,
                    model.x_class: x_spe_class_batch,  
                    model.x_gen_class: x_gen_class_batch,  #emb
                    model.y_cause: y_cause_batch, #label
                    model.y_cause4: y_cause4_batch,
                    model.y_cause3: y_cause3_batch,
                    model.y1_: y1_batch,  # specific
                    model.y2_: y2_batch,  # general
                    model.y_jg: y_jgs_batch,
                    model.keep_prob: keep_prob,
                    model.is_training: True
                }
                return feed_dict        
            
            
        
            def validation_step(mode, data_set, writer=None):
                """Evaluates model on a validation set"""
                cause_predict = []
                cause4_predict = []
                cause3_predict = []
                cause_truth = []
                cause4_truth = []
                cause3_truth = []

                a1_eval_acc_tk = [0.0] * args.topk
                a1_eval_pre_tk = [0.0] * args.topk
                a1_eval_rec_tk = [0.0] * args.topk
                a1_eval_F1_tk = [0.0] * args.topk
                a2_eval_acc_tk = [0.0] * args.topk
                a2_eval_pre_tk = [0.0] * args.topk
                a2_eval_rec_tk = [0.0] * args.topk
                a2_eval_F1_tk = [0.0] * args.topk

                a1_true_labels = []
                a2_true_labels = []
                a1_predicted_scores = []
                a2_predicted_scores = []
                a1_predicted_labels_ts = []
                a2_predicted_labels_ts = []
                a1_predicted_labels_tk = [[] for _ in range(args.topk)]
                a2_predicted_labels_tk = [[] for _ in range(args.topk)]

                plea_predict = []
                plea_truth = []

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                num_samples = len(data_set)
                div = num_samples % args.batch_size
                batch_num = num_samples // args.batch_size + 1 if div != 0 else num_samples // args.batch_size

                jg_eval_counter, jg_eval_loss, cl_eval_loss, cg_eval_loss, art_eval_loss = 0, 0.0, 0.0, 0.0, 0.0
                tf_data_set = tf.data.Dataset.from_generator(
                    lambda: data_set,
                    (tf.int32,  # fact, 
                     tf.int32, tf.int32, tf.int32, # plai defe plea 
                     tf.int32, tf.int32, tf.int32, tf.int32,  # cl, cl4 cl3, gl
                     tf.int32, tf.int32, tf.int32, tf.int32,  # sl, c, c4, c3
                     tf.int32, tf.int32, tf.int32)  # g, s, j
                ).padded_batch(args.batch_size,
                               padded_shapes=(
                                   tf.TensorShape([None]), 
                                   tf.TensorShape([None]), tf.TensorShape([None]),
                                   tf.TensorShape([None]),
                                   # plea, num , len
                                   tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                                   tf.TensorShape([None, None]),
                                   tf.TensorShape([None, None]),
                                   tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None]),
                                   tf.TensorShape([None]),
                                   tf.TensorShape([None]), tf.TensorShape([None]),
                                   tf.TensorShape([None])),
                               padding_values=(0, 0, 0, 0, 
                                               0, 0, 0, 0,
                                               0, 0, 0, 0,
                                               0, 0, 0))  # pad index 0
                tf_iterator = tf_data_set.make_one_shot_iterator()
                tf_one_batch = tf_iterator.get_next()

                '''ground_pleas_labels = []
                ground_length = []'''

                for batch_id in tqdm(range(batch_num)):
                    try:
                        bd_x,  \
                        bd_x_plai, bd_x_defe, bd_x_plea,  \
                        bd_cause_labelemb, bd_cause4_labelemb, bd_cause3_labelemb, \
                        bd_gen_labelemb, bd_spe_labelemb, \
                        bd_y_cause_, bd_y_cause4_, bd_y_cause3_, \
                        bd_y_general, bd_y_specific, bd_y_jgs = sess.run(tf_one_batch)
                        feed_dict = feed_data(bd_x, bd_x_plai, bd_x_defe, bd_x_plea,
                                              bd_cause_labelemb, bd_cause4_labelemb, bd_cause3_labelemb,
                                              bd_gen_labelemb, bd_spe_labelemb,
                                              bd_y_cause_, bd_y_cause4_, bd_y_cause3_,
                                              bd_y_specific, bd_y_general,
                                              bd_y_jgs, args.dropout_eval)
                        step, summaries, \
                        cause_acc, cause4_acc, cause3_acc, \
                        cause_scores, fourth_scores, third_scores, \
                        a1_scores, a2_scores, \
                        jgs_predict, \
                        jg_loss, cl_loss, cg_loss, art_loss, \
                        joint_loss = sess.run(
                            [model.global_step, validation_summary_op,
                            model.cause_acc, model.cause4_acc, model.cause3_acc,
                            model.pred_cause, model.pred_cause4, model.pred_cause3,
                            model.a1_scores, model.a2_scores,
                            model.pred_plea,
                            model.jg_loss, model.cause_local_loss, model.cause_global_loss, model.article_loss,
                            model.joint_loss], feed_dict)
                    except tf.errors.OutOfRangeError:  # 到达数据集末尾，如果需要继续使用迭代器 需要重新初始化
                        print("End of dataset")

                    
                    
                    for i in bd_y_cause_:
                        cause_truth.append(i)
                    for i in bd_y_cause4_:
                        cause4_truth.append(i)
                    for i in bd_y_cause3_:
                        cause3_truth.append(i)
                    for j in cause_scores:
                        cause_predict.append(j)
                    for j in fourth_scores:
                        cause4_predict.append(j)
                    for j in third_scores:
                        cause3_predict.append(j)
            
                    for i in bd_y_jgs:
                        plea_truth.append(i)
                    for j in jgs_predict:
                        plea_predict.append(j)    
                    
                    for i in bd_y_specific:
                        a1_true_labels.append(i)
                    for j in a1_scores:
                        a1_predicted_scores.append(j)
                    for i in bd_y_general:
                        a2_true_labels.append(i)
                    for j in a2_scores:
                        a2_predicted_scores.append(j)
                    # Predict by threshold
                    a1_batch_predicted_labels_ts = \
                        dh.get_onehot_label_threshold(scores=a1_scores, threshold=args.threshold)
                    for k in a1_batch_predicted_labels_ts:
                        a1_predicted_labels_ts.append(k)
                    a2_batch_predicted_labels_ts = \
                        dh.get_onehot_label_threshold(scores=a2_scores, threshold=args.threshold)
                    for k in a2_batch_predicted_labels_ts:
                        a2_predicted_labels_ts.append(k)

                    # Predict by topK
                    for top_num in range(args.topk):
                        a1_batch_predicted_labels_tk = dh.get_onehot_label_topk(scores=a1_scores, top_num=top_num + 1)
                        for i in a1_batch_predicted_labels_tk:
                            a1_predicted_labels_tk[top_num].append(i)
                    for top_num in range(args.topk):
                        a2_batch_predicted_labels_tk = dh.get_onehot_label_topk(scores=a2_scores, top_num=top_num + 1)
                        for i in a2_batch_predicted_labels_tk:
                            a2_predicted_labels_tk[top_num].append(i)

                    jg_eval_loss = jg_eval_loss + jg_loss
                    cl_eval_loss = cl_eval_loss + cl_loss
                    cg_eval_loss = cg_eval_loss + cg_loss
                    art_eval_loss = art_eval_loss + art_loss
                    jg_eval_counter = jg_eval_counter + 1
                    

                    if writer:
                        writer.add_summary(summaries, step)
                ###########Cause##############
                ##############################
                ##############################
             
                get_cause_score(cause4_truth, cause3_truth, cause_truth,
                                cause4_predict, cause3_predict, cause_predict, mode)
                
                ###########Articles##############
                ##############################
                ##############################
                get_articles_result(a1_true_labels, a1_predicted_labels_ts, a1_predicted_labels_tk,
                                    a2_true_labels, a2_predicted_labels_ts, a2_predicted_labels_tk,
                                    a1_eval_acc_tk, a1_eval_pre_tk, a1_eval_rec_tk, a1_eval_F1_tk,
                                    a2_eval_acc_tk, a2_eval_pre_tk, a2_eval_rec_tk, a2_eval_F1_tk,
                                    mode)
                ###########Judgment##############
                ##############################
                ##############################

                jg_acc, jg_pre, jg_rec, jg_f1 = get_result(plea_truth, plea_predict, mode)

                jg_eval_losses = float(jg_eval_loss / jg_eval_counter)
                cl_eval_losses = float(cl_eval_loss / jg_eval_counter)
                cg_eval_losses = float(cg_eval_loss / jg_eval_counter)
                art_eval_losses = float(art_eval_loss / jg_eval_counter)


                logger.info(f"Mode {mode}, Jg_Loss = {jg_eval_losses}, cause_local_Loss = {cl_eval_losses}, cause_global_Loss = {cg_eval_losses}, articles_Loss = {art_eval_losses}")

                return jg_eval_losses, jg_acc, jg_pre, jg_rec, jg_f1

            ##############train################
            ###################################
            plea_predict = []
            plea_truth = []
            total_batch = 0
            total_plea_loss = 0.
            total_jg_loss = 0.
            total_joint_loss = 0.
            total_pf_loss = 0.
            total_df_loss = 0.
            total_cause_loss = 0.
            total_c4_loss = 0.
            total_c3_loss = 0.
            
            train_num_samples = len(train_data)
            logger.info(f"Train samples:{train_num_samples}")
            div = train_num_samples % args.batch_size
            batch_num = train_num_samples // args.batch_size + 1 if div != 0 else train_num_samples // args.batch_size
            logger.info(f"batch_num:{batch_num}")
    

            train_data_set = tf.data.Dataset.from_generator(
                lambda: train_data,
                (tf.int32, tf.int32, tf.int32, tf.int32, 
                 # fact fact_len plai defe plea plea_num len
                 tf.int32, tf.int32, tf.int32, tf.int32,  # cl, cl4 cl3, gl
                 tf.int32, tf.int32, tf.int32, tf.int32,  # sl, c, c4, c3
                 tf.int32, tf.int32, tf.int32)  # g, s, j
            ).padded_batch(args.batch_size,
                           padded_shapes=(
                               tf.TensorShape([None]),  # fact
                               tf.TensorShape([None]), tf.TensorShape([None]),  # plai defe
                               tf.TensorShape([None]), # plea, 
                               tf.TensorShape([None, None]), tf.TensorShape([None, None]), tf.TensorShape([None, None]),
                               tf.TensorShape([None, None]),
                               tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None]),
                               tf.TensorShape([None]),
                               tf.TensorShape([None]), tf.TensorShape([None]), 
                               tf.TensorShape([None])),
                           padding_values=(0, 0, 0, 0, 
                                           0, 0, 0, 0,
                                           0, 0, 0, 0,
                                           0, 0, 0))  # pad index 0
            #train_data_set = train_data_set.shuffle(buffer_size=train_num_samples * 2)
            train_data_set = train_data_set.repeat()
            train_iterator = train_data_set.make_one_shot_iterator()
            train_one_batch = train_iterator.get_next()
            # Training loop. For each batch...

            for epoch in range(1, args.num_epochs+1):
                logger.info('Epoch: {0}'.format(epoch))

                for batch_id in tqdm(range(batch_num)):
                    try:
                        bt_x, bt_x_plai, bt_x_defe, bt_x_pleas,\
                        bt_cause_labelemb, bt_cause4_labelemb, bt_cause3_labelemb, \
                        bt_gen_labelemb, bt_spe_labelemb, \
                        bt_y_cause_, bt_y_cause4_, bt_y_cause3_, \
                        bt_y_general, bt_y_specific, bt_y_jgs \
                            = sess.run(train_one_batch)
                        
                        feed_dict = feed_data(bt_x, bt_x_plai, bt_x_defe, bt_x_pleas, 
                                          bt_cause_labelemb, bt_cause4_labelemb, bt_cause3_labelemb,
                                          bt_gen_labelemb, bt_spe_labelemb,
                                          bt_y_cause_, bt_y_cause4_, bt_y_cause3_,
                                          bt_y_specific, bt_y_general, bt_y_jgs, args.dropout) 
                        
                        step, _, _, lr, summaries, plea_loss, jg_loss, joint_loss, \
                        cause_global_loss, cause4_loss, cause3_loss, \
                        jg1_score, task_weight = sess.run(
                            [model.global_step, train_plea_op, train_cause_op, learning_rate, train_summary_op,
                            model.plea_loss, model.jg_loss, model.joint_loss,
                            model.cause_global_loss, model.c4_loss, model.c3_loss,
                            model.pred_plea, model.task_weight], feed_dict)
                        
                  
                        #####train evaluation
                        for i in bt_y_jgs:
                            plea_truth.append(i)
                        for j in jg1_score:
                            plea_predict.append(j) 
                    
                        #step, _, _, mi_pf_loss, mi_df_loss = sess.run([model.global_step, model.mi_pf_op, model.mi_df_op, model.mi_pf_loss, model.mi_df_loss],feed_dict)
                        
                        total_plea_loss += plea_loss
                        total_jg_loss += jg_loss
                        total_joint_loss += joint_loss
                        #total_pf_loss += mi_pf_loss
                        #total_df_loss += mi_df_loss
                        total_cause_loss += cause_global_loss
                        total_c4_loss += cause4_loss
                        total_c3_loss += cause3_loss
                    
                    except tf.errors.OutOfRangeError:  
                        logger.info("End of dataset")
                        break

                
                    if step % args.evaluate_steps == 0:  
                        logger.info(f"Epoch = {epoch}, Train: Plea_Loss = {total_plea_loss / (total_batch)}, Judg_Loss = {total_jg_loss / (total_batch)}, Joint_Loss = {total_joint_loss / (total_batch)}")
                        logger.info(f"Epoch = {epoch}, Train: Cause_loss = {total_cause_loss / (total_batch)}, cause4_loss = {total_c4_loss / (total_batch)}, cause3_Loss = {total_c3_loss / (total_batch)}")
                        jg_acc, jg_pre, jg_rec, jg_f1 = get_result(plea_truth, plea_predict, "train")
                        train_summary_writer.add_summary(summaries, step)
                        
                        #mi_pf_Loss = {total_pf_loss / (total_batch)}, df_Loss = {total_df_loss / (total_batch)}
                    

                    # best_saver.handle(jg_acc, sess, total_batch)#
                    total_batch += 1

                
                logger.info("epoch {0}, Validate:".format(epoch))
                jg1_eval_loss, jg_acc, jg1_eval_pre, jg1_eval_rec, jg1_eval_F1 = \
                    validation_step("validation", val_data, writer=validation_summary_writer)

                logger.info("epoch {0}, Test:".format(epoch))
                jg1_test_loss, jg_test_acc, jg_test_pre, jg_test_rec, jg_test_F1 = \
                    validation_step("test", test_data, None)
                
                saver.save(sess, checkpoint_prefix)
 
                
                
                if jg_test_acc > best_jg_acc:
                    # 保存最好结果
                    best_jg_epoch = epoch
                    best_jg_f1 = jg_test_F1
                    best_jg_acc = jg_test_acc
                    best_jg_pre = jg_test_pre
                    best_jg_rec = jg_test_rec
                    saver.save(sess, checkpoint_prefix)
                    best_saver.handle(best_jg_acc, sess, total_batch)

 

                '''if total_batch - last_improved > args.require_improvement:
                    print("No optimization for a long time, auto-stopping...")
                    break '''
                

            logger.info(f"epoch {best_jg_epoch}, best jg_acc: {best_jg_acc:.4f}, best_jg_pre: {best_jg_pre:.4f}, best_jg_rec: {best_jg_rec:.4f}, best_jg_f1: {best_jg_f1:.4f}")

        logger.info("All Done.")


if __name__ == '__main__':
    '''if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python civi.py [train / test]""")'''

    args = parser()
    logger = get_logger(f'./logs/{args.log}.log')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info(f"Args:{args}")
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.best_checkpoint_dir):
        os.makedirs(args.best_checkpoint_dir)

    logger.info("Loading training and validation data...")

    train_data, val_data, test_data = np_read_pre_new() # load numpy files
    '''train_data = train_data[:100]
    val_data = val_data[:100]
    test_data = test_data[:100]'''
    model = Article(args)
    cpee()
