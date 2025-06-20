import argparse
import gc
import os

from torch.optim import lr_scheduler, Adam, SGD
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW

from model_electra import TeacherModel, StudentModel, Bridge, PathClassifier
from dialogue_dataset import DialogueDataset, DiscourseGraph
from tqdm import tqdm

from utils import *
from utils import _get_clones

from sklearn.metrics import confusion_matrix
from collections import defaultdict
import pandas as pd

def compute_and_save_confusion_matrix(eval_matrix, csv_path="confusion_matrix.csv", json_path="confusion_matrix.json"):
    hypothesis_list = eval_matrix['hypothesis']
    reference_list = eval_matrix['reference']
    
    # Collect all unique relations and ensure they are strings
    all_relations = set()
    for hyp, ref in zip(hypothesis_list, reference_list):
        all_relations.update(map(str, hyp.values()))
        all_relations.update(map(str, ref.values()))
    all_relations.add("None")  # Include "None" for unmatched predictions or references
    relation_to_idx = {relation: idx for idx, relation in enumerate(sorted(all_relations))}
    idx_to_relation = {idx: relation for relation, idx in relation_to_idx.items()}
    
    # Prepare y_true and y_pred
    y_true = []
    y_pred = []
    
    for hyp, ref in zip(hypothesis_list, reference_list):
        # Reference pairs
        ref_set = set(ref.items())
        # Hypothesis pairs
        hyp_set = set(hyp.items())
        
        # Add matches from hypothesis
        for (x, y), relation in hyp.items():
            if (x, y) in ref:
                y_true.append(str(ref[(x, y)]))  # Ground truth
            else:
                y_true.append("None")  # False positive
            y_pred.append(str(relation))
        
        # Add missing matches from reference
        for (x, y), relation in ref.items():
            if (x, y) not in hyp:
                y_true.append(str(relation))  # Ground truth
                y_pred.append("None")  # False negative
    
    # Convert to numeric indices
    y_true_idx = [relation_to_idx[r] for r in y_true]
    y_pred_idx = [relation_to_idx[r] for r in y_pred]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(relation_to_idx.values()))
    
    # Save to CSV
    df = pd.DataFrame(cm, index=[idx_to_relation[i] for i in range(len(cm))],
                      columns=[idx_to_relation[i] for i in range(len(cm))])
    df.to_csv(csv_path, index_label="Actual / Predicted")
    
    # Save to JSON
    cm_dict = {
        idx_to_relation[row]: {
            idx_to_relation[col]: int(cm[row, col]) for col in range(len(cm))
        } for row in range(len(cm))
    }
    with open(json_path, 'w') as json_file:
        json.dump(cm_dict, json_file, indent=4)
    
    return cm_dict, csv_path, json_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--glove_vocab_path', type=str)
    parser.add_argument('--model_name_or_path', type=str, default='google/electra-small-discriminator') #change as per requirement
    parser.add_argument('--max_vocab_size', type=int, default=1000)
    parser.add_argument('--remake_dataset', action="store_true")
    parser.add_argument('--remake_tokenizer', action="store_true")
    parser.add_argument('--max_edu_dist', type=int, default=20)
    # model
    parser.add_argument('--glove_embedding_size', type=int, default=100)
    parser.add_argument('--path_hidden_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256) #default was 256 for electra, 768 for paraphrase
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--speaker', action='store_true')
    parser.add_argument('--valid_dist', type=int, default=4)
    parser.add_argument('--freeze_layer_num', type=int, default=9)
    # train
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--pretrained_model_learning_rate', type=float, default=1e-5)
    parser.add_argument('--epoches', type=int, default=10)
    parser.add_argument('--pool_size', type=int, default=1)
    parser.add_argument('--eval_pool_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--ratio', type=float, default=1.0)
    # save
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_path', type=str, default='student_model.pt')
    parser.add_argument('--teacher_model_path', type=str, default='teacher_model.pt')
    parser.add_argument('--overwrite', action="store_true")
    # other option
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--report_step', type=int, default=50)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--classify_loss', action='store_true')
    parser.add_argument('--classify_ratio', type=float, default=0.2)
    parser.add_argument('--distill_ratio', type=float, default=3.)
    parser.add_argument('--task', type=str)


    args = parser.parse_args()

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda")

    if not os.path.isdir(args.dataset_dir):
        os.mkdir(args.dataset_dir)

    # glove_tokenizer_path = os.path.join(args.dataset_dir, 'tokenizer.pt')
    # if args.remake_tokenizer:
    #     tokenizer = GloveTokenizer(args)
    #     torch.save(tokenizer, glove_tokenizer_path)
    # tokenizer = torch.load(glove_tokenizer_path)
    # pretrained_embedding = tokenizer.emb
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_data_file = os.path.join(args.dataset_dir, 'train.pt')
    eval_data_file = os.path.join(args.dataset_dir, 'eval.pt')
    test_data_file = os.path.join(args.dataset_dir, 'test.pt')

    if os.path.exists(train_data_file) and not args.remake_dataset:
        print('loading dataset..')
        train_dataset = torch.load(train_data_file)
        eval_dataset = torch.load(eval_data_file)
        relations, type2ids, id2types = train_dataset.relations, train_dataset.type2ids, train_dataset.id2types
        if not args.do_train:
            test_dataset = DialogueDataset(args=args, filename=args.test_file, tokenizer=tokenizer, mode='test')
            test_dataset.get_relations(relations, type2ids, id2types)
            test_dataset.get_discourse_graph()
            torch.save(test_dataset, test_data_file)
    else:
        train_dataset = DialogueDataset(args=args, filename=args.train_file, tokenizer=tokenizer, mode='train')
        eval_dataset = DialogueDataset(args=args, filename=args.eval_file, tokenizer=tokenizer, mode='eval')

        relations = train_dataset.relations | eval_dataset.relations
        type2ids, id2types = DialogueDataset.format_relations(relations)
        train_dataset.get_relations(relations, type2ids, id2types)
        eval_dataset.get_relations(relations, type2ids, id2types)

        train_dataset.get_discourse_graph()
        eval_dataset.get_discourse_graph()

        print('saving dataset..')
        torch.save(train_dataset, train_data_file)
        torch.save(eval_dataset, eval_data_file)

    args.relation_type_num = len(id2types)

    # print(relations)
    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)

    def train_collate_fn(examples):
        def pool(d):
            d = sorted(d, key=lambda x: x[6])
            edu_nums = [x[6] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))
            random.shuffle(buckets)

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]
                texts, pairs, graphs, lengths, speakers, turns, edu_nums, _ = zip(*batch)
                texts = DialogueDataset.nest_padding(texts)
                lengths = pad_sequence(lengths, batch_first=True, padding_value=1)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, pairs, graphs, lengths, speakers, turns, edu_nums

        return pool(examples)


    def eval_collate_fn(examples):
        texts, pairs, graphs, lengths, speakers, turns, edu_nums, ids = zip(*examples)
        texts = DialogueDataset.nest_padding(texts)
        lengths = pad_sequence(lengths, batch_first=True, padding_value=1)
        speakers = ints_to_tensor(list(speakers))
        turns = ints_to_tensor(list(turns))
        graphs = ints_to_tensor(list(graphs))
        edu_nums = torch.tensor(edu_nums)
        return texts, pairs, graphs, lengths, speakers, turns, edu_nums, ids


    if args.do_train:
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.pool_size, shuffle=True,
                                      collate_fn=train_collate_fn)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=args.eval_pool_size, shuffle=False,
                                     collate_fn=eval_collate_fn)

        # finetune last n layers
        if args.freeze_layer_num > 0:
            for param in pretrained_model.parameters():
                param.requires_grad = False
            for param in pretrained_model.encoder.layer[args.freeze_layer_num:].parameters():
                param.requires_grad = True

        if args.task=='distill':
            t_model = TeacherModel(params=args, pretrained_model=pretrained_model)
            t_state_dict=torch.load(args.teacher_model_path)
            t_model.load_state_dict(t_state_dict)
            model = StudentModel(params=args, pretrained_model=pretrained_model)

            bridges=_get_clones(Bridge(params=args), args.num_layers)
            param_groups=[{'params':[param for name, param in model.named_parameters() if
                            name.split('.')[0] != 'pretrained_model'], 'lr':args.learning_rate},
                          {'params':bridges.parameters(), 'lr':args.learning_rate}]
            t_model = t_model.to(args.device)
            bridge = bridges.to(args.device)
            t_model.eval()
        elif args.task=='student':
            model = StudentModel(params=args, pretrained_model=pretrained_model)
            param_groups = [{'params': [param for name, param in model.named_parameters() if
                            name.split('.')[0] != 'pretrained_model'], 'lr': args.learning_rate}]
        elif args.task=='teacher':
            model = TeacherModel(params=args, pretrained_model=pretrained_model)
            param_groups = [{'params': [param for name, param in model.named_parameters() if
                            name.split('.')[0] != 'pretrained_model'], 'lr': args.learning_rate}]
        if args.classify_loss:
            classifier = PathClassifier(params=args)
            param_groups.append({'params': classifier.parameters(),'lr':args.learning_rate})
            classifiers = classifier.to(args.device)
        param_groups.append({'params': filter(lambda p: p.requires_grad, model.pretrained_model.parameters()),
                             'lr': args.pretrained_model_learning_rate})
        optimizer = AdamW(params=param_groups, lr=args.learning_rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma) if args.gamma > 0 else None

        model = model.to(args.device)

        total_step = 0
        eval_result = {}
        accum_train_link_loss, accum_train_label_loss = 0, 0
        accum_distill_loss, accum_classify_loss = 0, 0
        accum_eval_loss = 0
        scheduler_step = 0
        best_eval_result = None
        stop_sign=0

        for epoch in range(args.epoches):
            print('{} epoch training..'.format(epoch + 1))
            print('dialogue model learning rate {:.4f}'.format(optimizer.param_groups[0]['lr']))
            model.train()
            for batch in tqdm(train_dataloader):
                for mini_batch in batch:
                    texts, pairs, graphs, lengths, speakers, turns, edu_nums = mini_batch
                    texts, graphs, lengths, speakers, turns, edu_nums = texts.cuda(), graphs.cuda(), lengths.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()

                    mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
                    optimizer.zero_grad()

                    if args.task=='teacher':
                        link_scores, label_scores, memory = model(texts, lengths, edu_nums, speakers,turns, graphs)
                    else:
                        link_scores, label_scores, memory = model(texts, lengths, edu_nums, speakers, turns)
                    link_loss, label_loss, negative_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask, negative=True)
                    link_loss = link_loss.mean()
                    label_loss = label_loss.mean()
                    loss = link_loss + label_loss + negative_loss*0.2
                    accum_train_link_loss += link_loss.item()
                    accum_train_label_loss += label_loss.item()

                    if args.task=='distill':
                        with torch.no_grad():
                            teacher_link_scores, teacher_label_scores, teacher_memory = t_model(texts, lengths, edu_nums, speakers, turns, graphs)
                        distill_loss=0
                        padding_mask =mask|mask.transpose(1,2)
                        for layer_pair in zip(memory, teacher_memory, bridges):
                            student_path, teacher_path, bridge = layer_pair[0], layer_pair[1], layer_pair[2]
                            distill_loss += nn.functional.mse_loss(bridge(student_path[padding_mask]),
                                                                      teacher_path[padding_mask],
                                                                      reduction='mean')
                        distill_loss*=args.distill_ratio
                        loss+=distill_loss
                        accum_distill_loss+=distill_loss.item()
                    if args.classify_loss:
                        classify_loss = 0
                        cls_mask = mask
                        cls_mask = cls_mask | cls_mask.transpose(1, 2)
                        cls_mask[:,0,:]=False
                        cls_mask[:,:,0]=False
                        target=graphs
                        target=target+target.transpose(1,2)
                        for student_path in memory[:-1]:
                            classify_loss += classifier(student_path, target, cls_mask)
                        classify_loss *= args.classify_ratio
                        loss += classify_loss
                        accum_classify_loss+=classify_loss.item()

                    loss.backward()
                    optimizer.step()

                if (total_step + 1) % args.report_step == 0:
                    if args.classify_loss:
                        print(
                            '\t{} step loss: {:.4f} {:.4f}, distill loss: {:.4f}, classify loss: {:.4f}'.format(total_step + 1,
                                                                                            accum_train_link_loss / args.report_step,
                                                                                            accum_train_label_loss / args.report_step,
                                                                                            accum_distill_loss / args.report_step,
                                                                                            accum_classify_loss / args.report_step))
                        accum_classify_loss = 0
                    else:
                        print(
                            '\t{} step loss: {:.4f} {:.4f}, distill loss: {:.4f}'.format(total_step + 1,
                                                                   accum_train_link_loss / args.report_step,
                                                                   accum_train_label_loss / args.report_step,
                                                                   accum_distill_loss / args.report_step))
                    accum_train_link_loss, accum_train_label_loss, accum_distill_loss = 0, 0, 0
                total_step += 1

                if scheduler and optimizer.param_groups[0]['lr'] > args.min_lr:
                    scheduler.step()
            print('{} epoch training done, begin evaluating..'.format(epoch + 1))
            accum_eval_link_loss, accum_eval_label_loss = [], []

            model = model.eval()

            eval_matrix = {
                'hypothesis': None,
                'reference': None,
                'edu_num': None
            }

            for batch in eval_dataloader:
                texts, pairs, graphs, lengths, speakers, turns, edu_nums, _ = batch
                texts, graphs, lengths, speakers, turns, edu_nums = texts.cuda(), graphs.cuda(), lengths.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()

                mask = get_mask(edu_nums + 1, args.max_edu_dist).cuda()

                with torch.no_grad():
                    if args.task=='teacher':
                        link_scores, label_scores, _ = model(texts, lengths, edu_nums, speakers, turns, graphs)
                    else:
                        link_scores, label_scores, _ = model(texts, lengths, edu_nums, speakers, turns)
                eval_link_loss, eval_label_loss = compute_loss(link_scores, label_scores, graphs, mask)
                accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
                accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))

                batch_size = link_scores.size(0)
                max_len = edu_nums.max()
                link_scores[~mask] = -1e9
                predicted_links = torch.argmax(link_scores, dim=-1)
                predicted_labels = torch.argmax(label_scores.reshape(-1, max_len + 1, args.relation_type_num)[
                                                    torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(
                                                        -1)].reshape(batch_size, max_len + 1, args.relation_type_num),
                                                dim=-1)
                predicted_links = predicted_links[:, 1:] - 1
                predicted_labels = predicted_labels[:, 1:]
                for i in range(batch_size):
                    hp_pairs = {}
                    step = 1
                    while step < edu_nums[i]:
                        link = predicted_links[i][step].item()
                        label = predicted_labels[i][step].item()
                        hp_pairs[(link, step)] = label
                        step += 1

                    predicted_result = {'hypothesis': hp_pairs,
                                        'reference': pairs[i],
                                        'edu_num': step}
                    record_eval_result(eval_matrix, predicted_result)

            print("---eval result---")
            a, b = zip(*accum_eval_link_loss)
            c, d = zip(*accum_eval_label_loss)
            eval_link_loss, eval_label_loss = sum(a) / sum(b), sum(c) / sum(d)
            print('eval loss : {:.4f} {:.4f}'.format(eval_link_loss, eval_label_loss))
            eval_loss = eval_link_loss + eval_label_loss

            f1_bi, f1_multi = tsinghua_F1(eval_matrix)
            print("link micro-f1 : {}\n"
                  "label micro-f1: {}".format(f1_bi, f1_multi))

            stop_sign += 1
            if best_eval_result is None or best_eval_result - eval_loss > 0:
                best_eval_result = eval_loss
                stop_sign=0
                if args.save_model:
                    print('saving model..')
                    torch.save(model.state_dict(), args.model_path)
            elif stop_sign+1>args.early_stop:
                break
        for k, v in eval_result.items():
            print(k, v)
    else:
        # eval_matrix = {
        #     'hypothesis': None,
        #     'reference': None,
        #     'edu_num': None
        # }

        # p_dict = {}
        # test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_pool_size, shuffle=False,
        #                              collate_fn=eval_collate_fn)

        # print('loading model state dict..')
        # if args.task=='teacher':
        #     model = TeacherModel(params=args, pretrained_model=pretrained_model)
        # else:
        #     model = StudentModel(params=args, pretrained_model=pretrained_model)
        # model.load_state_dict(torch.load(args.model_path))
        # model = model.to(args.device)
        # model = model.eval()

        # accum_eval_link_loss, accum_eval_label_loss = [], []
        # for batch in test_dataloader:
        #     texts, pairs, graphs, lengths, speakers, turns, edu_nums, ids = batch
        #     texts, graphs, lengths, speakers, turns, edu_nums = texts.cuda(), graphs.cuda(), lengths.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()

        #     mask = get_mask(edu_nums + 1, args.max_edu_dist).cuda()

        #     with torch.no_grad():
        #         if args.task == 'teacher':
        #             link_scores, label_scores, _ = model(texts, lengths, edu_nums, speakers, turns, graphs)
        #         else:
        #             link_scores, label_scores, _ = model(texts, lengths, edu_nums, speakers, turns)
        #     eval_link_loss, eval_label_loss = compute_loss(link_scores, label_scores, graphs, mask)
        #     accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
        #     accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))

        #     batch_size = link_scores.size(0)
        #     max_len = edu_nums.max()
        #     link_scores[~mask] = -1e9
        #     link_scores = nn.functional.softmax(link_scores, dim=-1).cpu()
        #     label_scores = nn.functional.softmax(label_scores, dim=-1).cpu()
        #     predicted_links = torch.argmax(link_scores, dim=-1)
        #     predicted_labels = torch.argmax(label_scores.reshape(-1, max_len + 1, args.relation_type_num)[
        #                                         torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(
        #                                             -1)].reshape(batch_size, max_len + 1, args.relation_type_num),
        #                                     dim=-1)
        #     predicted_links = predicted_links[:, 1:] - 1
        #     predicted_labels = predicted_labels[:, 1:]

        #     for i in range(batch_size):
        #         hp_pairs = {}
        #         step = 1
        #         edu_num = edu_nums[i]
        #         node_num = edu_num + 1
        #         save_links = torch.floor(link_scores[i, :node_num, :node_num] * 100)
        #         save_labels = torch.floor(label_scores[i, :node_num, :node_num] * 100)
        #         p_dict[ids[i]] = {'link_matrix': save_links, 'label_matrix': save_labels}
        #         while step < edu_num:
        #             link = predicted_links[i][step].item()
        #             label = predicted_labels[i][step].item()
        #             hp_pairs[(link, step)] = label
        #             step += 1

        #         predicted_result = {'hypothesis': hp_pairs,
        #                             'reference': pairs[i],
        #                             'edu_num': step}
        #         record_eval_result(eval_matrix, predicted_result)
        # print("---test result---")
        # a, b = zip(*accum_eval_link_loss)
        # c, d = zip(*accum_eval_label_loss)
        # eval_link_loss, eval_label_loss = sum(a) / sum(b), sum(c) / sum(d)
        # print('test loss : {:.4f} {:.4f}'.format(eval_link_loss, eval_label_loss))

        # # f1_bi, f1_multi = test_F1(eval_matrix)
        # # print("\n---eval result---"
        # #       "\nlink micro-f1 : {}"
        # #       "\nlabel micro-f1: {}".format(f1_bi, f1_multi))

        # f1_bi, f1_multi = tsinghua_F1(eval_matrix)
        # print("\n---eval result---"
        #       "\nlink micro-f1 : {}"
        #       "\nlabel micro-f1: {}\n".format(f1_bi, f1_multi))

        # # with open('predicted_result.pkl','wb')as file:
        # #     pickle.dump(eval_matrix, file)
        # # survey(eval_matrix, id2types)

        # # with open('turn_acc_list.pkl','wb') as file:
        # #     pickle.dump(p_value_prepare(eval_matrix), file)
        # # accuray_dist(eval_matrix)

        eval_matrix = {
            'hypothesis': None,
            'reference': None,
            'edu_num': None
        }

        p_dict = {}
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.eval_pool_size, shuffle=False,
                                    collate_fn=eval_collate_fn)

        print('loading model state dict..')
        if args.task == 'teacher':
            model = TeacherModel(params=args, pretrained_model=pretrained_model)
        else:
            model = StudentModel(params=args, pretrained_model=pretrained_model)
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(args.device)
        model = model.eval()

        accum_eval_link_loss, accum_eval_label_loss = [], []
        for batch in test_dataloader:
            texts, pairs, graphs, lengths, speakers, turns, edu_nums, ids = batch
            texts, graphs, lengths, speakers, turns, edu_nums = texts.cuda(), graphs.cuda(), lengths.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()

            mask = get_mask(edu_nums + 1, args.max_edu_dist).cuda()

            with torch.no_grad():
                if args.task == 'teacher':
                    link_scores, label_scores, _ = model(texts, lengths, edu_nums, speakers, turns, graphs)
                else:
                    link_scores, label_scores, _ = model(texts, lengths, edu_nums, speakers, turns)
            eval_link_loss, eval_label_loss = compute_loss(link_scores, label_scores, graphs, mask)
            accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
            accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))

            batch_size = link_scores.size(0)
            max_len = edu_nums.max()
            link_scores[~mask] = -1e9
            link_scores = nn.functional.softmax(link_scores, dim=-1).cpu()
            label_scores = nn.functional.softmax(label_scores, dim=-1).cpu()
            predicted_links = torch.argmax(link_scores, dim=-1)
            predicted_labels = torch.argmax(label_scores.reshape(-1, max_len + 1, args.relation_type_num)[
                                                torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(
                                                    -1)].reshape(batch_size, max_len + 1, args.relation_type_num),
                                            dim=-1)
            predicted_links = predicted_links[:, 1:] - 1
            predicted_labels = predicted_labels[:, 1:]

            # Save original predictions and targets
            with open("predicted_labels.txt", "w") as pred_file, open("target_labels.txt", "w") as target_file:
                for i in range(batch_size):
                    pred_file.write(str(predicted_labels[i].tolist()) + "\n")
                    target_file.write(str(pairs[i]) + "\n")

            # Apply relation mapping
            def map_relation(relation, merge_mapping, excluded_type):
                if relation == excluded_type:
                    return None
                for key, values in merge_mapping.items():
                    if relation in values:
                        return key
                return relation

            merge_mapping = {
                "Question_extension": ["Q_Elab", "Clarification_question"],
                "Conditional": ["Conditional", "Alternation"],
                "Continuation": ["Continuation", "Comment", "Elaboration", "Parallel", "Result", "Explanation"]
            }
            excluded_type = "Narration"

            for i in range(batch_size):
                hp_pairs = {}
                step = 1
                edu_num = edu_nums[i]
                node_num = edu_num + 1
                save_links = torch.floor(link_scores[i, :node_num, :node_num] * 100)
                save_labels = torch.floor(label_scores[i, :node_num, :node_num] * 100)
                p_dict[ids[i]] = {'link_matrix': save_links, 'label_matrix': save_labels}
                while step < edu_num:
                    link = predicted_links[i][step].item()
                    label = predicted_labels[i][step].item()
                    mapped_label = map_relation(label, merge_mapping, excluded_type)
                    if mapped_label is not None:
                        hp_pairs[(link, step)] = mapped_label
                    step += 1

                # Transform reference pairs using the same mapping
                ref_pairs = {}
                for (link, step), label in pairs[i].items():
                    mapped_label = map_relation(label, merge_mapping, excluded_type)
                    if mapped_label is not None:
                        ref_pairs[(link, step)] = mapped_label

                predicted_result = {'hypothesis': hp_pairs,
                                    'reference': ref_pairs,
                                    'edu_num': step}
                record_eval_result(eval_matrix, predicted_result)
        print("---test result---")
        a, b = zip(*accum_eval_link_loss)
        c, d = zip(*accum_eval_label_loss)
        eval_link_loss, eval_label_loss = sum(a) / sum(b), sum(c) / sum(d)
        print('test loss : {:.4f} {:.4f}'.format(eval_link_loss, eval_label_loss))

        # Compute F1 scores with the new mapping
        f1_bi, f1_multi = tsinghua_F1(eval_matrix)
        print("\n---eval result---"
            "\nlink micro-f1 : {}"
            "\nlabel micro-f1: {}\n".format(f1_bi, f1_multi))
        
        confusion_matrix_dict, csv_file, json_file = compute_and_save_confusion_matrix(eval_matrix)
        print(f"Confusion matrix saved to:\nCSV: {csv_file}\nJSON: {json_file}")



