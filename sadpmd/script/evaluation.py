from data.Dataloader import *
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json
import pandas as pd

RELATION_TYPE_MAP = {
    "Background": 0,
    "Acknowledgment": 1,
    "Continuation": 2,
    "dia-continuation": 3,
    "Question-answer complaint Pair": 4,
    "Conditional": 5,
    "Question-answer_pair": 6,
    "Question Extension": 7,
    "Correction": 8,
    "Contrast": 9
}

def build_confusion_matrix(gold_file, predict_file, merge_mapping, excluded_type):
    gold_instances = read_corpus(gold_file)
    predict_instances = read_corpus(predict_file)
    assert len(gold_instances) == len(predict_instances), "Mismatch in the number of instances."

    # Initialize the confusion matrix
    num_classes = len(RELATION_TYPE_MAP)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for g_instance, p_instance in zip(gold_instances, predict_instances):
        # Extract gold and predicted relations
        g_rels = get_rels(g_instance)
        p_rels = get_rels(p_instance)

        # Iterate over gold relations and compare with predicted relations
        for g_rel in g_rels:
            # Split the gold relation to extract the relation type
            _, _, g_type = g_rel.rsplit("##", 2)
            gold_index = RELATION_TYPE_MAP.get(g_type, -1)
            
            if gold_index == -1:
                continue  # Skip if relation type is unknown
            
            # Check if this relation exists in predicted relations
            if g_rel in p_rels:
                pred_index = gold_index  # Correct prediction
            else:
                # Find the predicted type for this relation, if any
                # This assumes a closest match or the predicted relation exists with a different type
                pred_index = next((RELATION_TYPE_MAP.get(rel.rsplit("##", 2)[-1], -1) for rel in p_rels if rel.rsplit("##", 2)[0:2] == g_rel.rsplit("##", 2)[0:2]), -1)
                if pred_index == -1:
                    continue  # Skip if predicted type is also unknown

            # Update the confusion matrix
            confusion_matrix[gold_index][pred_index] += 1

    return confusion_matrix

def save_confusion_matrix_as_csv(conf_matrix, output_file):
    import pandas as pd
    labels = list(RELATION_TYPE_MAP.keys())
    df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    df.to_csv(output_file, index=True, mode="a")
    print(f"Confusion matrix saved to {output_file}")


#for training
def evaluation(gold_file, predict_file, merge_mapping, excluded_type):

    gold_instances = read_corpus(gold_file)
    predict_instances = read_corpus(predict_file)
    assert len(gold_instances) == len(predict_instances)
    uas_metric, las_metric = Metric(), Metric()
    for g_instance, p_instance in zip(gold_instances, predict_instances):
        g_arcs = get_arcs(g_instance)
        p_arcs = get_arcs(p_instance)
        uas_metric.correct_label_count += len(g_arcs & p_arcs)
        uas_metric.overall_label_count += len(g_arcs)
        uas_metric.predicated_label_count += len(p_arcs)

        # g_rels = get_rels(g_instance, merge_mapping, excluded_type)
        # p_rels = get_rels(p_instance, merge_mapping, excluded_type)

        g_rels = get_rels(g_instance)
        p_rels = get_rels(p_instance)

        las_metric.correct_label_count += len(g_rels & p_rels)
        las_metric.overall_label_count += len(g_rels)
        las_metric.predicated_label_count += len(p_rels)

        # assert len(g_arcs) == len(g_rels)
    print("UAS:")
    uas_metric.print()
    print("LAS:")
    las_metric.print()

    #confusion matrix
    conf_matrix = build_confusion_matrix(gold_file, predict_file, merge_mapping=None, excluded_type=None)
    save_confusion_matrix_as_csv(conf_matrix, "comumdr_confusion_matrix_divyaksh.csv")
    
    return las_metric.getAccuracy() 

def get_arcs(instance):
    arcs = set()
    for cur_y, relations in enumerate(instance.real_relations):
        if len(relations) == 0:
            y = str(cur_y)
            x = str(-1)
            arc = y + "##" + x
            arcs.add(arc)
        else:
            for relation in relations:
                y = str(relation['y'])
                x = str(relation['x'])
                arc = y + "##" + x
                arcs.add(arc)
    return arcs

def get_rels(instance):
    rels = set()
    for cur_y, relations in enumerate(instance.real_relations):
        if len(relations) == 0:
            y = str(cur_y)
            x = str(-1)
            rel = y + "##" + x + "##" + '<root>'
            rels.add(rel)
        else:
            for relation in relations:
                y = str(relation['y'])
                x = str(relation['x'])
                rel = y + "##" + x + "##" + relation['type']
                rels.add(rel)
    return rels

# def get_rels(instance, merge_mapping, excluded_type):
#     rels = set()
#     for cur_y, relations in enumerate(instance.real_relations):
#         if len(relations) == 0:
#             y = str(cur_y)
#             x = str(-1)
#             rel = y + "##" + x + "##" + '<root>'
#             rels.add(rel)
#         else:
#             for relation in relations:
#                 y = str(relation['y'])
#                 x = str(relation['x'])
#                 rel_type = merge_relation_type(relation['type'], merge_mapping, excluded_type)
#                 if rel_type:
#                     rel = y + "##" + x + "##" + rel_type
#                     rels.add(rel)
#     return rels

# #added 
# def merge_relation_type(rel_type, merge_mapping, excluded_type):
#     """
#     Merge and map the relation type based on the merge_mapping.

#     :param rel_type: Original relation type
#     :param merge_mapping: Dictionary defining merge rules
#     :param excluded_type: Type to exclude
#     :return: Merged relation type or None if excluded
#     """
#     # print(f"relation type is {rel_type}")
#     if rel_type == excluded_type:
#         return 99
#     for key, values in merge_mapping.items():
#         if rel_type in values:
#             return key
#     return rel_type  # If not in mapping, keep original


class Metric:
    def __init__(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def reset(self):
        self.overall_label_count = 0
        self.correct_label_count = 0
        self.predicated_label_count = 0

    def bIdentical(self):
        if self.predicated_label_count == 0:
            if self.overall_label_count == self.correct_label_count:
                return True
            return False
        else:
            if self.overall_label_count == self.correct_label_count and \
                    self.predicated_label_count == self.correct_label_count:
                return True
            return False

    def getAccuracy(self):
        if self.overall_label_count + self.predicated_label_count == 0:
            return 1.0
        if self.predicated_label_count == 0:
            return self.correct_label_count*1.0 / self.overall_label_count
        else:
            return self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)

    def print(self):
        if self.predicated_label_count == 0:
            print("Accuracy:\tP=" + str(self.correct_label_count) + '/' + str(self.overall_label_count))
        else:
            print("Recall:\tP=" + str(self.correct_label_count) + "/" + str(self.overall_label_count) + "=" + str(self.correct_label_count*1.0 / self.overall_label_count), end=",\t")
            print("Accuracy:\tP=" + str(self.correct_label_count) + "/" + str(self.predicated_label_count) + "=" + str(self.correct_label_count*1.0 / self.predicated_label_count), end=",\t")
            print("Fmeasure:\t" + str(self.correct_label_count*2.0 / (self.overall_label_count + self.predicated_label_count)))





