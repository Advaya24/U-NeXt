import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
# from utils import DiceLoss
from utils import test_single_volume, DiceLoss
from torchvision import transforms
import wandb
import gc

def test(model, testloader, epoch_num, args, length):
    test_save_path = os.path.join('../predictions', args.model, args.model_type)
    os.makedirs(test_save_path, exist_ok=True)
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        h, w = sampled_batch["image"].size()[3:]
        image, label, case_name = sampled_batch["image"].cuda(), sampled_batch["label"].cuda(), sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                    test_save_path=test_save_path, case=case_name, z_spacing=1)
        metric_list += np.array(metric_i)

        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / length
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    wandb.log(
        data = {
            'train_loss': epoch_train_loss,
            'metric_list': metric_list,
            'performance': performance,
            'mean_hd95': mean_hd95
        },
        step = epoch_num
    )

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    wandb.init(project='unext-synapse')
    if 'UNext' in args.model:
        wandb.run.name = f'{args.model}_{args.model_type}'
    else:
        wandb.run.name = f'{args.model}'
    # wandb.tensorboard.patch(tensorboardX=True)
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    db_test = Synapse_dataset(base_dir='../data/Synapse/test_vol_h5', split="test_vol", list_dir=args.list_dir)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        epoch_train_loss = 0.0
        epoch_train_ce = 0.0
        epoch_train_dice = 0.0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            # loss = 0.5 * loss_ce + 0.5 * loss_dice
            loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            epoch_train_loss += loss.item()
            epoch_train_dice += loss_dice.item()
            epoch_train_ce += loss_ce.item()

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 10  # int(max_epoch/6)

        wandb.log(
            data = {
                'train_loss': epoch_train_loss,
                'train_CE': epoch_train_ce,
                'train_dice': epoch_train_dice
            },
            step = epoch_num
        )

        if epoch_num > int(max_epoch / 4) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            # test(model, testloader, epoch_num, args, len(db_test))
            break

    # test(model, testloader, args.max_epochs, args, len(db_test))
    writer.close()
    return "Training Finished!"