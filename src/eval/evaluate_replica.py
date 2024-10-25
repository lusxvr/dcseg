import numpy as np
import files
import sys
import os
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../third_party/opennerf')))

import opennerf.datasets.replica as replica

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def eval_semantics(scene, masks, pairing):

    pr_files = []  # predicted files
    gt_files = []  # ground truth files
    pr_files.append(f'{files.RESULT_PATH}pr_semantics/semantics_{pairing}_{scene}_{masks}.txt')
    gt_files.append(f'{files.OPENNERF_PATH}datasets/replic_gt_semantics/semantic_labels_{scene}.txt')

    confusion = np.zeros([replica.num_classes, replica.num_classes], dtype=np.ulonglong)
    
    print('evaluating', len(pr_files), 'scans...')
    for i in range(len(pr_files)):
        evaluate_scan(pr_files[i], gt_files[i], confusion)
        sys.stdout.write("\rscans processed: {}".format(i+1))
        sys.stdout.flush()

    class_ious = {}

    for i in range(replica.num_classes):
        label_name = replica.class_names_reduced[i]
        label_id = i
        class_ious[label_name] = get_iou(label_id, confusion)

        print('classes \t IoU \t Acc')
        print('----------------------------')
        for i in range(replica.num_classes):
            label_name = replica.class_names_reduced[i]
            print('{0:<14s}: {1:>5.2%}   {2:>6.2%}'.format(label_name, class_ious[label_name][0], class_ious[label_name][1]))

    iou_values = np.array([i[0] for i in class_ious.values()])
    acc_values = np.array([i[1] for i in class_ious.values()])
    print()
    print(f'mIoU: \t {np.mean(iou_values):.2%}')
    print(f'mAcc: \t {np.mean(acc_values):.2%}')
    print()

    split_mIoU = []
    split_mAcc = []
    for i, split in enumerate(['head', 'comm', 'tail']):
        print(f'{split}: \t {np.mean(iou_values[17 * i:17 * (i + 1)]):.2%}')
        print(f'{split}: \t {np.mean(acc_values[17 * i:17 * (i + 1)]):.2%}')
        print('---')

        split_mIoU.append(np.mean(iou_values[17 * i:17 * (i + 1)]))
        split_mAcc.append(np.mean(acc_values[17 * i:17 * (i + 1)]))

    metrics = {}
    metrics['Total mIoU'] = np.mean(iou_values)
    metrics['Total mAcc'] = np.mean(acc_values)
    metrics['Head mIoU'] = split_mIoU[0]
    metrics['Head mAcc'] = split_mAcc[0]
    metrics['Comm mIoU'] = split_mIoU[1]
    metrics['Comm mAcc'] = split_mAcc[1]
    metrics['Tail mIoU'] = split_mIoU[2]
    metrics['Tail mAcc'] = split_mAcc[2]
    metrics['Class IoUs / Acc'] = class_ious

    with open(os.path.join(files.RESULT_PATH, f'metrics/{pairing}_{scene}_{masks}'), 'w') as f:
        json.dump(metrics, f, indent=4)


def evaluate_scan(pr_file, gt_file, confusion):

    pr_ids = np.array(process_txt(pr_file), dtype=np.int64)
    gt_file_contents = np.array(process_txt(gt_file)).astype(np.int64)
    gt_ids = np.vectorize(replica.map_to_reduced.get)(gt_file_contents)

    # sanity checks
    if not pr_ids.shape == gt_ids.shape:
        print(f'number of predicted values does not match number of vertices: {pr_file}')
    for (gt_val, pr_val) in zip(gt_ids, pr_ids):
        if gt_val == replica.num_classes:
            continue
        confusion[gt_val][pr_val] += 1


def get_iou(label_id, confusion):
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    fp = np.longlong(confusion[:, label_id].sum()) - tp
    denom = float(tp + fp + fn)
    if denom == 0:
        return (0, 0) #float('nan')
    iou = tp / denom

    if tp==0 and fn==0:
        return (iou, 0)
    acc = tp / float(tp + fn)
    
    return (iou, acc)

def main(scene, masks, pairing):
    eval_semantics(scene, masks, pairing)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Opennerf Evaluation Pipeline on Replica.")
    parser.add_argument('--scene', required=True, help="Scene to evaluate.")
    parser.add_argument('--masks', required=True, help="Masks from scene to evaluate.")
    parser.add_argument('--pairing', required=True, help="Type of mask pairing to evaluate. Either \"matching\" or \"assignment\".")

    args = parser.parse_args()
    main(args.scene, args.masks, args.pairing)