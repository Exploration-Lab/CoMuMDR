import sys
sys.path.extend(["../../", "../", "./"])
import pickle
import random
import time
import argparse
from data.Config import *
from modules.DialogDP import *
from modules.Optimizer import *
from modules.Decoder import *
from script.evaluation import *
from modules.GlobalEncoder import *
from data.BertTokenHelper import *
from modules.BertModelTune import *
from modules.SPEncoder import SPEncoder

from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler


def train(train_instances, dev_instances, test_instances, parser, vocab, config, tokenizer, merge_mapping, excluded_type ):
    global_step = 0
    best_dev_las = 0
    batch_num = int(np.ceil(len(train_instances) / float(config.train_batch_size)))

    bert_param = list(parser.global_encoder.bert_extractor.parameters())

    parser_param = \
        list(parser.global_encoder.mlp_words.parameters()) + \
        list(parser.global_encoder.rescale.parameters()) + \
        list(parser.global_encoder.edu_GRU.parameters()) + \
        list(parser.sp_encoder.parameters()) + \
        list(parser.state_encoder.parameters()) + \
        list(parser.decoder.parameters())

    model_param = [{'params': bert_param, 'lr': config.bert_learning_rate},
                   {'params': parser_param, 'lr': config.learning_rate}]
    
    optimizer = Optimizer(model_param, config)
    scaler = GradScaler()

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_arc_correct, overall_arc_total, overall_rel_correct = 0, 0, 0
        for onebatch in data_iter(train_instances, batch_size=config.train_batch_size, shuffle=True):
            parser.train()
            batch_input_ids, batch_token_type_ids, batch_attention_mask, token_lengths = \
                batch_bert_variable(onebatch, config, tokenizer)
            breakpoint()
            batch_sp = batch_sp_variable(onebatch, vocab)
            edu_lengths, arc_masks = batch_data_variable(onebatch, vocab)
            feats = batch_feat_variable(onebatch, vocab)
            gold_arcs, gold_rels = batch_label_variable(onebatch, vocab)
            with autocast():
                parser.forward(batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_sp, token_lengths,
                               edu_lengths, arc_masks, feats)
                loss = parser.compute_loss(gold_arcs, gold_rels)
                loss = loss / config.update_every
            scaler.scale(loss).backward()
            # loss.backward()
            loss_value = loss.item()

            arc_correct, arc_total, rel_correct = parser.compute_accuracy(gold_arcs, gold_rels)
            
            overall_rel_correct += rel_correct
            overall_arc_correct += arc_correct
            overall_arc_total += arc_total
            uas = overall_arc_correct * 100.0 / overall_arc_total
            las = overall_rel_correct * 100.0 / overall_arc_total

            during_time = float(time.time() - start_time)
            print("Step:%d, uas:%.2f, las:%.2f Iter:%d, batch:%d, time:%.2f, loss:%.2f" \
                  % (global_step, uas, las, iter, batch_iter, during_time, loss_value))

            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                scaler.unscale_(optimizer.optim)
                nn.utils.clip_grad_norm_(bert_param + parser_param, max_norm=config.clip)

                scaler.step(optimizer.optim)
                scaler.update()
                optimizer.schedule()

                optimizer.zero_grad()
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                with torch.no_grad():
                    predict(dev_instances, parser, vocab, config, tokenizer, config.dev_file + "." + str(global_step))
                print("Dev:")
                dev_las = evaluation(config.dev_file, config.dev_file + "." + str(global_step), merge_mapping, excluded_type)

                with torch.no_grad():
                    predict(test_instances, parser, vocab, config, tokenizer, config.test_file + "." + str(global_step))
                print("Test:")
                evaluation(config.test_file, config.test_file + "." + str(global_step), merge_mapping, excluded_type)

                if dev_las > best_dev_las:
                    print("Exceed best uas F-score: history = %.2f, current = %.2f" % (best_dev_las, dev_las))
                    best_dev_las = dev_las

                    if config.save_after >= 0 and iter >= config.save_after:

                        dp_model = {
                            "mlp_words": parser.global_encoder.mlp_words.state_dict(),
                            "rescale": parser.global_encoder.rescale.state_dict(),
                            "edu_GRU": parser.global_encoder.edu_GRU.state_dict(),
                            "sp_encoder": parser.sp_encoder.state_dict(),
                            "state_encoder": parser.state_encoder.state_dict(),
                            "decoder": parser.decoder.state_dict()
                        }

                        torch.save(dp_model, config.save_model_path)
                        save_bert_path = config.save_bert_path
                        global_encoder.bert_extractor.bert.save_pretrained(save_bert_path)
                        tokenizer.tokenizer.save_pretrained(save_bert_path)
                        print('Saving model to ', config.save_dir)               


def predict(instances, parser, vocab, config, tokenizer, outputFile):
    start = time.time()
    parser.eval()
    pred_instances = []
    for onebatch in data_iter(instances, batch_size=config.test_batch_size, shuffle=False):
        edu_lengths, arc_masks = batch_data_variable(onebatch, vocab)
        dialog_feats = batch_feat_variable(onebatch, vocab)
        batch_sp = batch_sp_variable(onebatch, vocab)
        batch_input_ids, batch_token_type_ids, batch_attention_mask, token_lengths = \
            batch_bert_variable(onebatch, config, tokenizer)
        with autocast():
            pred_arcs, pred_rels = parser.forward(
                batch_input_ids, batch_token_type_ids, batch_attention_mask, batch_sp, token_lengths,
                edu_lengths, arc_masks, dialog_feats)
        for batch_index, (arcs, rels) in enumerate(zip(pred_arcs, pred_rels)):
            instance = onebatch[batch_index]
            length = len(instance.EDUs)
            relation_list = []
            for idx in range(length):
                if idx == 0 or idx == 1: continue
                y = idx - 1
                x = int(arcs[idx] - 1)
                type = vocab.id2rel(rels[idx])
                relation = dict()
                relation['y'] = y
                relation['x'] = x
                relation['type'] = type
                relation_list.append(relation)
            dialog = onebatch[batch_index]
            pred_dialog = dict()
            pred_dialog['edus'] = dialog.original_EDUs
            pred_dialog['id'] = dialog.id
            pred_dialog['relations'] = relation_list 
            pred_instances.append(pred_dialog)
    out_f = open(outputFile, 'w', encoding='utf8')
    json.dump(pred_instances, out_f)
    out_f.close()
    print("Doc num: %d,  parser time = %.2f " % (len(instances), float(time.time() - start)))


if __name__ == '__main__':
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    ### seed
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    # torch version
    print("Torch Version: ", torch.__version__)

    # gpu state
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: ", torch.backends.cudnn.enabled)

    # args
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    train_instances = read_corpus(config.train_file, config.max_edu_num)
    dev_instances = read_corpus(config.dev_file)
    test_instances = read_corpus(config.test_file)

    print("train dialog num: ", len(train_instances))
    print("dev dialog num: ", len(dev_instances))
    print("test dialog num: ", len(test_instances))

    vocab = create_vocab(train_instances, config.max_vocab_size)
    torch.set_num_threads(args.thread)

    tok_helper = BertTokenHelper(config.bert_dir)
    bert_extractor = BertExtractor(config.bert_dir, config, tok_helper)

    ### use gpu or cpu
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    global_encoder = GlobalEncoder(vocab, config, bert_extractor)
    state_encoder = StateEncoder(vocab, config)
    sp_encoder = SPEncoder(vocab, config)
    decoder = Decoder(vocab, config)

    # print(global_encoder)
    print(state_encoder)
    print(sp_encoder)
    print(decoder)

    if config.use_cuda:
        torch.backends.cudnn.enabled = True

        global_encoder.cuda()
        state_encoder.cuda()
        sp_encoder.cuda()
        decoder.cuda()
    parser = DialogDP(global_encoder, state_encoder, sp_encoder, decoder, config)

    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    merge_mapping = {
        "Question_extension": ["Q_Elab", "Clarification_question"],
        "Conditional": ["Conditional", "Alternation"],
        "Continuation": ["Continuation", "Comment", "Elaboration", "Parallel", "Result", "Explanation"]
    }
    excluded_type = "Narration"

    train(train_instances, dev_instances, test_instances, parser, vocab, config, tok_helper, merge_mapping, excluded_type)
