import torch
import os
from utils import test_single_volume
from unext import UNeXt
from resunet import UNetWithResnet50Encoder
import argparse
import logging
import random
import numpy as np
from datasets.dataset_synapse import Synapse_dataset
from torch.utils.data import DataLoader

def test(model, testloader, args, length):
    test_save_path = os.path.join('../predictions', args.model, args.model_type)
    os.makedirs(test_save_path, exist_ok=True)
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        image, label, case_name = sampled_batch["image"].cuda(), sampled_batch["label"].cuda(), sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                    test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / length
    for i in range(1, args.num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))


parser = argparse.ArgumentParser()
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--model', default="UNext",
                        choices=["UNext", "Resunet"],
                        help='Network model to be trained (default: UNeXt)')
parser.add_argument('--model_type', default='convnext_tiny', 
                        choices=['convnext_xlarge_in22k', 'convnext_large_in22k', 'convnext_base_in22k', 'convnext_small', 'convnext_tiny'])
parser.add_argument('--path', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    if args.model == 'UNext':
        model = UNeXt(args.num_classes, args.model_type)
    else:
        model = UNetWithResnet50Encoder(args.num_classes)
    model.load_state_dict(torch.load(args.path))
    model = model.to('cuda:0')

    db_test = Synapse_dataset(base_dir='../data/Synapse/test_vol_h5', split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    print("The length of test set is: {}".format(len(db_test)))
    test(model, testloader, args, len(db_test))