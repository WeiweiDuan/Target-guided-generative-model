import argparse
from evaluation import precision_recall_f1

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--annotation_dir", type=str)
parser.add_argument("--pred_dir", type=str)
parser.add_argument("--obj_name", type=str)
parser.add_argument("--loc_idx", type=str)
parser.add_argument("--grid_size", type=int)
parser.add_argument("--image_size", type=int)
args = parser.parse_args()

precision, recall, f1 = precision_recall_f1(args.data_dir, args.annotation_dir, args.pred_dir, args.obj_name, args.loc_idx, args.grid_size, args.image_size)
print('precision, recall, f1 = ', precision, recall, f1)

