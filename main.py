import argparse
import os
import torch
from dataset import get_loader, ImageData
from solver import Solver


def main(config):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # if config.mode == 'train':
    train_loader = get_loader(config.train_img_path, config.train_label_path, config.img_size, config.batch_size,
                                filename=config.train_file, num_thread=config.num_thread)
    test_dataset = ImageData(config.test_img_path, config.test_label_path, filename=config.test_file, test=True, require_name=True)

    run = 0
    while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
    os.mkdir("%s/run-%d" % (config.save_fold, run))
    os.mkdir("%s/run-%d/logs" % (config.save_fold, run))
    # os.mkdir("%s/run-%d/images" % (config.save_fold, run))
    os.mkdir("%s/run-%d/models" % (config.save_fold, run))
    config.save_fold = "%s/run-%d" % (config.save_fold, run)
    
    train = Solver(train_loader, test_dataset, config)
    if config.mode == 'train':
        train.train()
    elif config.mode == 'test':
        train.test(100)
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    data_root = '/home/hanqi/dataset/'
    vgg_path = './weights/vgg16_feat.pth'
    # # -----ECSSD dataset-----
    # train_path = os.path.join(data_root, 'ECSSD/images')
    # label_path = os.path.join(data_root, 'ECSSD/ground_truth_mask')
    #
    # val_path = os.path.join(data_root, 'ECSSD/val_images')
    # val_label = os.path.join(data_root, 'ECSSD/val_ground_truth_mask')
    # test_path = os.path.join(data_root, 'ECSSD/test_images')
    # test_label = os.path.join(data_root, 'ECSSD/test_ground_truth_mask')
    # # -----MSRA-B dataset-----
    train_img_path = os.path.join(data_root, 'DUTS/DUTS-TR/')
    train_label_path = os.path.join(data_root, 'DUTS/DUTS-TR/')
    test_img_path = os.path.join(data_root, 'DUTS/DUTS-TE/DUTS-TE-Image')
    test_label_path = os.path.join(data_root, 'DUTS/DUTS-TE/DUTS-TE-Mask')
    train_file = os.path.join(data_root, 'DUTS/DUTS-TR/train_pair.lst')
    test_file = os.path.join(data_root, 'DUTS/DUTS-TE/test.lst')
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=256)  # 256
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1997)

    # Training settings
    parser.add_argument('--step', type=str, default='50,80')
    parser.add_argument('--update', type=int, default=8)
    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--train_img_path', type=str, default=train_img_path)
    parser.add_argument('--train_label_path', type=str, default=train_label_path)
    parser.add_argument('--train_file', type=str, default=train_file)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)  # 8
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--epoch_val', type=int, default=10)
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    # Testing settings
    parser.add_argument('--test_img_path', type=str, default=test_img_path)
    parser.add_argument('--test_label_path', type=str, default=test_label_path)
    parser.add_argument('--test_file', type=str, default=test_file)
    parser.add_argument('--model', type=str, default='./weights/final.pth')
    parser.add_argument('--test_fold', type=str, default='./results/test')
    parser.add_argument('--use_crf', type=bool, default=False)

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
