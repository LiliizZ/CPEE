import argparse
import pprint



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=int, default=0.0002)#3e-4
    parser.add_argument('--lr_cause', type=int, default=1e-5)#3e-4
    parser.add_argument('--decay_steps', type=int, default=1500)#700
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--dropout_eval', type=int, default=1.0)
    parser.add_argument('--decay_rate', type=int, default=0.95)
    parser.add_argument('--norm_ratio', type=int, default=1.25)#1.25
    parser.add_argument('--threshold', type=int, default=0.3)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--num_outputs1', type=int, default=76)#77
    parser.add_argument('--num_outputs2', type=int, default=95)#123
    parser.add_argument('--num_cause4', type=int, default=5)
    parser.add_argument('--num_cause3', type=int, default=6)
    parser.add_argument('--num_cause', type=int, default=11)
    parser.add_argument('--num_judg', type=int, default=3)
    parser.add_argument('--num_checkpoints', type=int, default=10)
    parser.add_argument('--evaluate_steps', type=int, default=1000)

    parser.add_argument('--require_improvement', type=int, default=10000)
    parser.add_argument('--vocab_size', type=int, default=100000)
    parser.add_argument('--article_vocab_size', type=int, default=10000)
    parser.add_argument('--cause_vocab_size', type=int, default=100)
    parser.add_argument('--maxlen', type=int, default=200)
    parser.add_argument('--max_len_fact', type=int, default=200)
    parser.add_argument('--max_len_plai', type=int, default=100)
    parser.add_argument('--max_len_defe', type=int, default=50)
    parser.add_argument('--maxlen1', type=int, default=100)
    parser.add_argument('--maxlen2', type=int, default=20)

    

    parser.add_argument('--embedding_type', type=int, default=1)
    parser.add_argument('--embed_word_dim', type=int, default=256)
    parser.add_argument('--embed_word_cause', type=int, default=200)
    parser.add_argument('--lstm_hidden_size', type=int, default=256)
    parser.add_argument('--fc_hidden_size', type=int, default=256)
    parser.add_argument('--attention_unit_size', type=int, default=200)


    parser.add_argument('--rnn', type=str, default='lstm')
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints2/')
    parser.add_argument('--best_checkpoint_dir', type=str, default='./bestcheckpoints2/')
    parser.add_argument('--summary', type=str, default='summaries2')
    parser.add_argument('--log', type=str, default='2')
    
    

    parser.add_argument('--gpu', type=str, default='1', help='which gpus to use')
    args = parser.parse_args()
    
    
    # pprint.PrettyPrinter().pprint(args.__dict__)
    return args