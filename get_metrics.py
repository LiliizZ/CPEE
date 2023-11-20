from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from utils import data_helpers as dh
import time

import logging
logger = logging.getLogger(__file__)

def get_cause_score(y_cause4_, y_cause3_, y_cause_,
                                fourth_scores, third_scores, cause_scores, mode):
    cause4_acc = accuracy_score(np.argmax(y_cause4_, 1), fourth_scores)
    cause4_eval_pre = precision_score(y_true=np.argmax(y_cause4_, 1),
                                        y_pred=fourth_scores, average="macro")
    cause4_eval_rec = recall_score(y_true=np.argmax(y_cause4_, 1),
                                    y_pred=fourth_scores, average="macro")
    cause4_eval_F1 = f1_score(y_true=np.argmax(y_cause4_, 1),
                                y_pred=fourth_scores, average="macro")

    cause3_acc = accuracy_score(np.argmax(y_cause3_, 1), third_scores)
    cause3_eval_pre = precision_score(y_true=np.argmax(y_cause3_, 1),
                                        y_pred=third_scores, average="macro")
    cause3_eval_rec = recall_score(y_true=np.argmax(y_cause3_, 1),
                                    y_pred=third_scores, average="macro")
    cause3_eval_F1 = f1_score(y_true=np.argmax(y_cause3_, 1),
                                y_pred=third_scores, average="macro")

    cause_acc = accuracy_score(np.argmax(y_cause_, 1), cause_scores)
    cause_eval_pre = precision_score(y_true=np.argmax(y_cause_, 1),
                                        y_pred=cause_scores, average="macro")
    cause_eval_rec = recall_score(y_true=np.argmax(y_cause_, 1),
                                    y_pred=cause_scores, average="macro")
    cause_eval_F1 = f1_score(y_true=np.argmax(y_cause_, 1),
                                y_pred=cause_scores, average="macro")


    msg1 = "Set {0:>6} Cause Predict: Accuracy {1:>7.2%}, Precision {2:>7.2%}, Recall {3:>7.2%}, F1 {4:>7.2%}"
    logger.info(msg1.format(mode, cause_acc, cause_eval_pre, cause_eval_rec, cause_eval_F1))


    msg2 = "Set {0:>6} layer 4 Predict: Accuracy {1:>7.2%}, Precision {2:>7.2%}, Recall {3:>7.2%}, F1 {4:>7.2%}"
    logger.info(msg2.format(mode, cause4_acc, cause4_eval_pre, cause4_eval_rec, cause4_eval_F1))
    

    msg3 = "Set {0:>6} layer 3 Predict: Accuracy {1:>7.2%}, Precision {2:>7.2%}, Recall {3:>7.2%}, F1 {4:>7.2%}"
    logger.info(msg3.format(mode, cause3_acc, cause3_eval_pre, cause3_eval_rec, cause3_eval_F1))
    

    return cause_acc, cause_eval_pre, cause_eval_rec, cause_eval_F1, \
            cause4_acc, cause4_eval_pre, cause4_eval_rec, cause4_eval_F1, \
            cause3_acc, cause3_eval_pre, cause3_eval_rec, cause3_eval_F1



def get_articles_result(a1_true_labels, a1_predicted_labels_ts, a1_predicted_labels_tk,
                                    a2_true_labels, a2_predicted_labels_ts, a2_predicted_labels_tk,
                                    a1_eval_acc_tk, a1_eval_pre_tk, a1_eval_rec_tk, a1_eval_F1_tk,
                                    a2_eval_acc_tk, a2_eval_pre_tk, a2_eval_rec_tk, a2_eval_F1_tk,
                                    mode):

    # Calculate Precision & Recall & F1
    a1_eval_acc_ts = accuracy_score(y_true=np.array(a1_true_labels),
                                    y_pred=np.array(a1_predicted_labels_ts))
    a1_eval_pre_ts = precision_score(y_true=np.array(a1_true_labels),
                                        y_pred=np.array(a1_predicted_labels_ts), average='micro')
    a1_eval_rec_ts = recall_score(y_true=np.array(a1_true_labels),
                                    y_pred=np.array(a1_predicted_labels_ts), average='micro')
    a1_eval_F1_ts = f1_score(y_true=np.array(a1_true_labels),
                                y_pred=np.array(a1_predicted_labels_ts), average='micro')

    a2_eval_acc_ts = accuracy_score(y_true=np.array(a2_true_labels),
                                    y_pred=np.array(a2_predicted_labels_ts))
    a2_eval_pre_ts = precision_score(y_true=np.array(a2_true_labels),
                                        y_pred=np.array(a2_predicted_labels_ts), average='micro')
    a2_eval_rec_ts = recall_score(y_true=np.array(a2_true_labels),
                                    y_pred=np.array(a2_predicted_labels_ts), average='micro')
    a2_eval_F1_ts = f1_score(y_true=np.array(a2_true_labels),
                                y_pred=np.array(a2_predicted_labels_ts), average='micro')

     # Predict by threshold
    msg_a11 = "set {0:>6} a1 Predict by threshold: Accuracy {1:>7.2%}, Precision {2:>7.2%}, Recall {3:>7.2%}, F1 {4:>7.2%}"
    logger.info(msg_a11.format(mode, a1_eval_acc_ts, a1_eval_pre_ts, a1_eval_rec_ts, a1_eval_F1_ts))
  

    msg_a12 = "set {0:>6} a2 Predict by threshold: Accuracy {1:>7.2%}, Precision {2:>7.2%}, Recall {3:>7.2%}, F1 {4:>7.2%}"
    logger.info(msg_a12.format(mode, a2_eval_acc_ts, a2_eval_pre_ts, a2_eval_rec_ts, a2_eval_F1_ts))

    
    '''for top_num in range(2):
        a1_eval_acc_tk[top_num] = accuracy_score(y_true=np.array(a1_true_labels),
                                                    y_pred=np.array(a1_predicted_labels_tk[top_num]))

        a1_eval_pre_tk[top_num] = precision_score(y_true=np.array(a1_true_labels),
                                                    y_pred=np.array(a1_predicted_labels_tk[top_num]),
                                                    average='micro')
        a1_eval_rec_tk[top_num] = recall_score(y_true=np.array(a1_true_labels),
                                                y_pred=np.array(a1_predicted_labels_tk[top_num]),
                                                average='micro')
        a1_eval_F1_tk[top_num] = f1_score(y_true=np.array(a1_true_labels),
                                            y_pred=np.array(a1_predicted_labels_tk[top_num]),
                                            average='micro')

    for top_num in range(2):
        a2_eval_acc_tk[top_num] = accuracy_score(y_true=np.array(a2_true_labels),
                                                    y_pred=np.array(a2_predicted_labels_tk[top_num]))
        a2_eval_pre_tk[top_num] = precision_score(y_true=np.array(a2_true_labels),
                                                    y_pred=np.array(a2_predicted_labels_tk[top_num]),
                                                    average='micro')
        a2_eval_rec_tk[top_num] = recall_score(y_true=np.array(a2_true_labels),
                                                y_pred=np.array(a2_predicted_labels_tk[top_num]),
                                                average='micro')
        a2_eval_F1_tk[top_num] = f1_score(y_true=np.array(a2_true_labels),
                                            y_pred=np.array(a2_predicted_labels_tk[top_num]),
                                            average='micro')

    # Predict by topK
    logger.info("Predict by topK:")
    for top_num in range(2):
        msg_ak1 = "set {0:>6} a1 Top{1}: Accuracy {2:>7.2%}, Precision {3:>7.2%}, Recall {4:>7.2%}, F1 {5:>7.2%}"
        logger.info(msg_ak1.format(mode, top_num + 1, a1_eval_acc_tk[top_num], a1_eval_pre_tk[top_num],
                                    a1_eval_rec_tk[top_num],
                                    a1_eval_F1_tk[top_num]))


    for top_num in range(2):
        msg_ak2 = "set {0:>6} a2 Top{1}: Accuracy {2:>7.2%}, Precision {3:>7.2%}, Recall {4:>7.2%}, F1 {5:>7.2%}"
        logger.info(msg_ak2.format(mode, top_num + 1, a2_eval_acc_tk[top_num], a2_eval_pre_tk[top_num],
                                    a2_eval_rec_tk[top_num],
                                    a2_eval_F1_tk[top_num]))'''

    return a1_eval_acc_ts, a2_eval_acc_ts




def get_result(target, preds, mode):
    target = np.argmax(target, 1)
    acc = accuracy_score(target, preds)
    macro_f1 = f1_score(target, preds, average="macro")
    macro_precision = precision_score(target, preds, average="macro")
    macro_recall = recall_score(target, preds, average="macro")
    # f_test.write("step".format(current_step) + mode + "\n")

    msg = "All judgment set {0:>6}: Accuracy {1:>7.2%}, Precision {2:>7.2%}, Recall {3:>7.2%}, F1 {4:>7.2%}"

    logger.info(msg.format(mode, acc, macro_precision, macro_recall, macro_f1))


    '''from collections import Counter
    sorted_target = sorted(Counter(target).items())
    sorted_preds = sorted(Counter(preds).items())
    print("ground:", sorted_target)
    print("pred  :", sorted_preds)
    logger.info("ground: (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_target[0][1], sorted_target[1][1],
                                                                    sorted_target[2][1]))
    logger.info("pred  : (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_preds[0][1], sorted_preds[1][1],
                                                                    sorted_preds[2][1]))

    '''
    '''target_names = ['驳回诉请', '部分支持', "支持诉请"]
    
    print(classification_report(target, preds, target_names=target_names, digits=4))
    logger.info(classification_report(target, preds, target_names=target_names, digits=4))
    f_test.write(classification_report(target, preds, target_names=target_names, digits=4) + "\n")'''

    return acc, macro_precision, macro_recall, macro_f1



