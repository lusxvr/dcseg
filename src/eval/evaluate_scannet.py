# Evaluates semantic label task
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_label.py --scan_path [path to scan data] --output_file [output file]

# python imports
import math
import os, sys, argparse
import inspect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import files

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)
try:
    from itertools import izip
except ImportError:
    izip = zip

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from helpers import util, util_3d

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
UNKNOWN_ID = np.max(VALID_CLASS_IDS) + 1


def evaluate_scan(pred_file, gt_file, confusion):
    try:
        pred_ids = util_3d.load_ids(pred_file)
    except Exception as e:
        util.print_error('unable to load ' + pred_file + ': ' + str(e))
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))
    # sanity checks
    if not pred_ids.shape == gt_ids.shape:
        util.print_error('%s: number of predicted values does not match number of vertices' % pred_file, user_fault=True)
    for (gt_val,pred_val) in izip(gt_ids.flatten(),pred_ids.flatten()):
        gt_val = int(gt_val/1000)
        if gt_val not in VALID_CLASS_IDS:
            continue
        if pred_val not in VALID_CLASS_IDS:
            pred_val = UNKNOWN_ID
        confusion[gt_val][pred_val] += 1


def get_iou(label_id, confusion):
    if not label_id in VALID_CLASS_IDS:
        return float('nan')
    # #true positives
    tp = np.longlong(confusion[label_id, label_id])
    # #false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # #false positives
    not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return (0, tp, 0, 0, 0) #float('nan')
    iou = tp / denom

    if tp==0 and fn==0:
         return (iou, 0, 0, 0, 0)
    acc = tp / float(tp + fn)

    return (iou, tp, denom, acc, fn)


def write_result_file(confusion, ious, total_iou, total_acc, filename):
    with open(filename, 'w') as f:
        f.write('total IoU: {0:>5.3f}\n'.format(total_iou))
        f.write('total Acc: {0:>5.3f}\n'.format(total_acc))
        f.write('iou scores\n')
        for i in range(len(VALID_CLASS_IDS)):
            label_id = VALID_CLASS_IDS[i]
            label_name = CLASS_LABELS[i]
            iou = ious[label_name][0]
            f.write('{0:<14s}({1:<2d}): {2:>5.3f}\n'.format(label_name, label_id, iou))
        f.write('\nconfusion matrix\n')
        f.write('\t\t\t')
        for i in range(len(VALID_CLASS_IDS)):
            #f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
            f.write('{0:<8d}'.format(VALID_CLASS_IDS[i]))
        f.write('\n')
        for r in range(len(VALID_CLASS_IDS)):
            f.write('{0:<14s}({1:<2d})'.format(CLASS_LABELS[r], VALID_CLASS_IDS[r]))
            for c in range(len(VALID_CLASS_IDS)):
                f.write('\t{0:>5.3f}'.format(confusion[VALID_CLASS_IDS[r],VALID_CLASS_IDS[c]]))
            f.write('\n')
    print('wrote results to', filename)


def evaluate(pred_files, gt_files, output_file):
    max_id = UNKNOWN_ID
    confusion = np.zeros((max_id+1, max_id+1), dtype=np.ulonglong)

    print('evaluating', len(pred_files), 'scans...')
    for i in range(len(pred_files)):
        evaluate_scan(pred_files[i], gt_files[i], confusion)
        sys.stdout.write("\rscans processed: {}".format(i+1))
        sys.stdout.flush()
    print('')

    class_ious = {}
    all_tp = 0
    all_points = 0
    all_fn = 0
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        class_ious[label_name] = get_iou(label_id, confusion)
        all_tp += class_ious[label_name][1]
        all_points += class_ious[label_name][2]
        all_fn += class_ious[label_name][4]
    # print(class_ious)

    total_iou = all_tp / all_points
    total_acc = all_tp / float(all_tp + all_fn)
    print(f"(all_tp / all_points) = {all_tp} / {all_points}")
    print(f'Total IoU: {total_iou:.4f}')
    print(f'Total Acc: {total_acc:.4f}')
    print('classes          IoU 	    TP/Total     Acc')
    print('---------------------------------------------')
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        #print('{{0:<14s}: 1:>5.3f}'.format(label_name, class_ious[label_name][0]))
        print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})  {4:3f}'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2], class_ious[label_name][3]))
    write_result_file(confusion, class_ious, total_iou, total_acc, output_file)


def main(scene, masks, pairing):
    pred_files = []  # predicted files
    gt_files = []  # ground truth files
    pred_files.append(f'{files.RESULT_PATH}pr_semantics/semantics_{pairing}_{scene}_{masks}.txt')
    gt_files.append(f'{files.DATA_PATH}scannet_gt_scenes/train/{scene}.txt')
    
    # pred_files = [f for f in os.listdir(opt.pred_path) if f.endswith('.txt') and f != 'semantic_label_evaluation.txt']
    # gt_files = []
    # if len(pred_files) == 0:
    #     util.print_error('No result files found.', user_fault=True)
    # for i in range(len(pred_files)):
    #     gt_file = os.path.join(opt.gt_path, pred_files[i])
    #     if not os.path.isfile(gt_file):
    #         util.print_error('Result file {} does not match any gt file'.format(pred_files[i]), user_fault=True)
    #     gt_files.append(gt_file)
    #     pred_files[i] = os.path.join(opt.pred_path, pred_files[i])

    # evaluate
    output_file = f"{files.RESULT_PATH}metrics/{pairing}_{scene}_{masks}.txt"
    evaluate(pred_files, gt_files, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Evaluation Pipeline on Scannet.")
    parser.add_argument('--scene', required=True, help="Scene to evaluate.")
    parser.add_argument('--masks', required=True, help="Masks from scene to evaluate.")
    parser.add_argument('--pairing', required=True, help="Type of mask pairing to evaluate. Either \"matching\" or \"assignment\".")

    args = parser.parse_args()
    main(args.scene, args.masks, args.pairing)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pred_path', required=True, help='path to directory of predicted .txt files')
    # parser.add_argument('--gt_path', required=True, help='path to gt files')
    # parser.add_argument('--output_file', default='', help='output file [default: pred_path/semantic_label_evaluation.txt]')
    # opt = parser.parse_args()

    # if opt.output_file == '':
    #     opt.output_file = os.path.join(opt.pred_path, 'semantic_label_evaluation.txt')


    # main()
