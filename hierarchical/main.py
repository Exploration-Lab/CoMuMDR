import tensorflow as tf
import numpy as np
import os, random, time
from Model import Model
from utils import load_data, build_vocab, preview_data, get_batches
from sklearn.metrics import confusion_matrix
import itertools

if not os.environ.has_key('CUDA_VISIBLE_DEVICES'): 
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('train', False, 'train model')
tf.flags.DEFINE_integer('display_interval', 500, 'step interval to display information')
tf.flags.DEFINE_boolean('show_predictions', False, 'show predictions in the test stage')
tf.flags.DEFINE_string('word_vector', 'glove/glove.6B.100d.txt', 'word vector')
tf.flags.DEFINE_string('embedding_model', 'FacebookAI/xlm-roberta-base', 'embedding model') #added
tf.flags.DEFINE_string('prefix', 'dev', 'prefix for storing model and log')
tf.flags.DEFINE_integer('vocab_size', 1000, 'vocabulary size')
tf.flags.DEFINE_integer('max_edu_dist', 20, 'maximum distance between two related edus') 
tf.flags.DEFINE_integer('dim_embed_word', 100, 'dimension of word embedding') #change to 100 when using glove and to 768 when using paraphrase
tf.flags.DEFINE_integer('dim_embed_relation', 100, 'dimension of relation embedding') #change to 100 when using glove and to 768 when using paraphrase
tf.flags.DEFINE_integer('dim_feature_bi', 4, 'dimension of binary features')
tf.flags.DEFINE_boolean('use_structured', True, 'use structured encoder')
tf.flags.DEFINE_boolean('use_speaker_attn', True, 'use speaker highlighting mechanism')
tf.flags.DEFINE_boolean('use_shared_encoders', False, 'use shared encoders')
tf.flags.DEFINE_boolean('use_random_structured', False, 'use random structured repr.')
tf.flags.DEFINE_integer('num_epochs', 50, 'number of epochs')
tf.flags.DEFINE_integer('num_units', 256, 'number of hidden units')
tf.flags.DEFINE_integer('num_layers', 1, 'number of RNN layers in encoders')
tf.flags.DEFINE_integer('num_relations', 16, 'number of relation types')
tf.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.flags.DEFINE_float('keep_prob', 0.5, 'probability to keep units in dropout')
tf.flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
tf.flags.DEFINE_float('learning_rate_decay', 0.98, 'learning rate decay factor')


from sklearn.metrics import confusion_matrix
from collections import Counter

from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(test_batches, model, map_relations, vocab):
    """
    Computes confusion matrix for the entire test dataset by matching (x, y, type).

    Args:
        test_batches: List of test batches.
        model: Trained model to make predictions.
        map_relations: Dictionary mapping relation indices to relation names.
        vocab: Vocabulary used in the dataset.

    Returns:
        Confusion matrix as a numpy array.
    """
    map_relations_inv = {v: k for k, v in map_relations.items()}
    true_relations = []
    predicted_relations = []

    for batch in test_batches:
        if len(batch[0]['edus']) == 1:
            continue
        
        # Get model predictions
        ops = model.step(batch)

        # Collect true relations from the batch
        for relation in batch[0]["relations"]:  # Assuming 'relations' is in the batch
            true_relations.append((relation["x"], relation["y"], relation["type"]))
        
        # Collect predicted relations from the model output
        for relation in ops[-1]:  # Assuming ops[-1] contains the predictions
            predicted_relations.append((relation[0], relation[1], relation[2]))

    # Match on (x, y, type) and extract the `type` for confusion matrix calculation
    true_types = [relation[2] for relation in true_relations]
    predicted_types = [
        relation[2] if relation in true_relations else -1  # Use -1 for unmatched predictions
        for relation in predicted_relations
    ]

    # Ensure true_types and predicted_types have the same length
    if len(predicted_types) < len(true_types):
        predicted_types += [-1] * (len(true_types) - len(predicted_types))  # Fill with unmatched
    elif len(predicted_types) > len(true_types):
        true_types += [-1] * (len(predicted_types) - len(true_types))  # Fill with unmatched

    # Calculate confusion matrix
    class_names = list(map_relations.keys())
    cm = confusion_matrix(true_types, predicted_types, labels=list(map_relations.values()) + [-1])
    
    # Print confusion matrix
    print("Confusion Matrix:")
    for i, row in enumerate(cm):
        if i < len(class_names):
            print("Class {}: {}".format(class_names[i], "\t".join(map(str, row))))
        else:
            print("Unmatched: {}".format("\t".join(map(str, row))))
    
    return cm




    
def get_summary_sum(s, length):
    loss_bi, loss_multi = s[0] / length, s[1] / length
    prec_bi, recall_bi = s[4] * 1. / s[3], s[4] * 1. / s[2]
    f1_bi = 2 * prec_bi * recall_bi / (prec_bi + recall_bi)
    prec_multi, recall_multi = s[5] * 1. / s[3], s[5] * 1. / s[2]
    f1_multi = 2 * prec_multi * recall_multi / (prec_multi + recall_multi)
    return [loss_bi, loss_multi, f1_bi, f1_multi]    
    
map_relations = {}
# data_train = load_data('./data/Molweni/train.json', map_relations)
# data_test = load_data('./data/Molweni/test.json', map_relations)

data_train = load_data('../comumdr_data/train.json', map_relations)
data_test = load_data('../comumdr_data/test.json', map_relations)

vocab, embed = build_vocab(data_train)
print( 'Dataset sizes: %d/%d' % (len(data_train), len(data_test)))
model_dir, log_dir = FLAGS.prefix + '_model', FLAGS.prefix + '_log'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
with sess.as_default():
    model = Model(sess, FLAGS, embed, data_train)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step_inc_op = global_step.assign(global_step + 1)    
    epoch = tf.Variable(0, name='epoch', trainable=False)
    epoch_inc_op = epoch.assign(epoch + 1)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=None, pad_step_number=True)
    
    summary_list = ['loss_bi', 'loss_multi', 'f1_bi', 'f1_multi']
    summary_num = len(summary_list)
    len_output_feed = 6

    if FLAGS.train:
        if tf.train.get_checkpoint_state(model_dir):
            print ('Reading model parameters from %s' % model_dir)
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        else:
            print( 'Created model with fresh parameters')
            sess.run(tf.global_variables_initializer())  
            model.initialize(vocab)          
            
        print( 'Trainable variables:')
        for var in tf.trainable_variables():
            print (var)

        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
        test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))
        summary_placeholders = [tf.placeholder(tf.float32) for i in range(summary_num)]
        summary_op = [tf.summary.scalar(summary_list[i], summary_placeholders[i]) for i in range(summary_num)]
        
        train_batches = get_batches(data_train, FLAGS.batch_size)
        test_batches = get_batches(data_test, FLAGS.batch_size)
        
        best_test_f1 = [0] * 2
        while epoch.eval() < FLAGS.num_epochs:
            epoch_inc_op.eval()
            summary_steps = 0
            
            random.shuffle(train_batches)
            start_time = time.time()
            s = np.zeros(len_output_feed)
            
            for batch in train_batches:
                ops = model.step(batch, is_train=True)
                    
                for i in range(len_output_feed):
                    s[i] += ops[i]
                        
                summary_steps += 1
                global_step_inc_op.eval()
                global_step_val = global_step.eval()         
                if global_step_val % FLAGS.display_interval == 0:
                    print ('epoch %d, global step %d (%.4fs/step):' % (
                        epoch.eval(), global_step_val, 
                        (time.time() - start_time) * 1. / summary_steps
                    ))
                    summary_sum = get_summary_sum(s, summary_steps)
                    for k in range(summary_num):
                        print( '  train %s: %.5lf' % (summary_list[k], summary_sum[k]))
                    print( '  best test f1:', best_test_f1[0], best_test_f1[1])
                        
            summary_sum = get_summary_sum(s, len(train_batches))            
            summaries = sess.run(summary_op, feed_dict=dict(zip(summary_placeholders, summary_sum)))
            for s in summaries:
                train_writer.add_summary(summary=s, global_step=epoch.eval())                        
            print( 'epoch %d (learning rate %.5lf)' % \
                (epoch.eval(), model.learning_rate.eval()))
            for k in range(summary_num):
                print ('  train %s: %.5lf' % (summary_list[k], summary_sum[k])      )             
            
            s = np.zeros(len_output_feed)
            random.seed(0)
            for batch in test_batches:
                ops = model.step(batch)
                for i in range(len_output_feed):
                    s[i] += ops[i]
            summary_sum = get_summary_sum(s, len(test_batches))
            summaries = sess.run(summary_op, feed_dict=dict(zip(summary_placeholders, summary_sum)))
            for s in summaries:
                test_writer.add_summary(summary=s, global_step=epoch.eval())            
            for k in range(summary_num):
                print( '  test %s: %.5lf' % (summary_list[k], summary_sum[k]) )
                
            if summary_sum[-1] > best_test_f1[1]:
                best_test_f1[0] = summary_sum[-2]
                best_test_f1[1] = summary_sum[-1]
            
            print( '  best test f1:', best_test_f1[0], best_test_f1[1])
            
            model.learning_rate_decay_op.eval()             
            
            saver.save(sess, '%s/checkpoint' % model_dir, global_step=epoch.eval())                                            
    else:
        print ('Reading model parameters from %s' % model_dir )
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        test_batches = get_batches(data_test, 1, sort=False)

        cm = compute_confusion_matrix(test_batches, model, map_relations, vocab)

        s = np.zeros(len_output_feed)
        random.seed(0)
        idx = 0
        for k, batch in enumerate(test_batches):
            if len(batch[0]['edus']) == 1: 
                continue    
            ops = model.step(batch)
            for i in range(len_output_feed):
                s[i] += ops[i]
            if FLAGS.show_predictions:
                idx = preview_data(batch, ops[-1], map_relations, vocab, idx) 
        summary_sum = get_summary_sum(s, len(test_batches))
    
        print( 'Test:')
        for k in range(summary_num):
            print( '  test %s: %.5lf' % (summary_list[k], summary_sum[k])   )

        ''''modified code for convin data eval'''
        # print('Reading model parameters from %s' % model_dir)
        # saver.restore(sess, tf.train.latest_checkpoint(model_dir))

        # test_batches = get_batches(data_test, 1, sort=False)

        # s = np.zeros(len_output_feed)
        # random.seed(0)
        # idx = 0

        # # Lists to store true labels and predictions
        # true_labels = []
        # predicted_labels = []

        # for k, batch in enumerate(test_batches):
        #     if len(batch[0]['edus']) == 1:
        #         continue

        #     ops = model.step(batch)
        #     for i in range(len_output_feed):
        #         s[i] += ops[i]

        #     # Collect true labels and predictions for the confusion matrix
        #     true_labels.extend(batch['true_labels'])  # Assuming 'true_labels' contains the ground truth
        #     predicted_labels.extend(ops[-1])  # Assuming ops[-1] contains the predictions

        #     if FLAGS.show_predictions:
        #         idx = preview_data(batch, ops[-1], map_relations, vocab, idx)

        # summary_sum = get_summary_sum(s, len(test_batches))

        # print('Test:')
        # for k in range(summary_num):
        #     print('  test %s: %.5lf' % (summary_list[k], summary_sum[k]))
        # # Generate and display the confusion matrix
        # class_names = list(map_relations.keys())  # Assuming map_relations contains the mapping of class indices to names
        # cm = confusion_matrix(true_labels, predicted_labels, labels=list(map_relations.values()))
        # print_confusion_matrix(cm, class_names)