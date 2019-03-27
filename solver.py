import torch
import os
import cv2
import numpy as np
from tools.summaries import TensorboardSummary
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.backends import cudnn
from torchvision import transforms
from dssnet import build_model, weights_init, Bwdist
from loss import Loss
from tools.visual import Viz_visdom
from tqdm import tqdm


class Solver(object):
    def __init__(self, train_loader, test_dataset, config):
        self.train_loader = train_loader
        self.test_dataset = test_dataset
        self.config = config
        self.beta = 0.3  # for max F_beta metric
        # inference: choose the side map (see paper)
        self.select = [1, 2, 3, 6]
        # self.device = torch.device('cpu')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        self.update = config.update
        self.step = config.step
        #modified by hanqi 
        self.summary = TensorboardSummary("%s/logs/" % config.save_fold)
        self.writer = self.summary.create_summary()
        self.visual_save_fold = config.save_fold
        if self.config.cuda:
            cudnn.benchmark = True
            # self.device = torch.device('cuda:0')
        if config.visdom:
            self.visual = Viz_visdom("DSS", 1)
        self.build_model()
        if self.config.pre_trained: self.net.module.load_state_dict(torch.load(self.config.pre_trained))
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.t_transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.round(x))
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
            
        else:
            self.net.module.load_state_dict(torch.load(self.config.model)["state_dict"])
            self.net.eval()
            # self.test_output = open("%s/test.txt" % config.test_fold, 'w')
            

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            if p.requires_grad: num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = torch.nn.DataParallel(build_model()).cuda()
        if self.config.mode == 'train': self.loss = Loss().cuda()
        self.net.train()
        self.net.apply(weights_init)
        if self.config.load == '': self.net.module.base.load_state_dict(torch.load(self.config.vgg))
        if self.config.load != '': self.net.module.load_state_dict(torch.load(self.config.load))
        self.optimizer = Adam(self.net.parameters(), self.config.lr)
        self.print_network(self.net, 'DSS')

    # update the learning rate
    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10.0

    # evaluate MAE (for test or validation phase)
    def eval_mae(self, y_pred, y):
        return torch.abs(y_pred - y).mean()

    # TODO: write a more efficient version
    # get precisions and recalls: threshold---divided [0, 1] to num values
    def eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
        return prec, recall



    # test phase: using origin image size, evaluate MAE and max F_beta metrics
    def test(self, num, use_crf=False, epoch=None):
        if use_crf: from tools.crf_process import crf
        avg_mae, img_num = 0.0, 0.0
        avg_prec, avg_recall = torch.zeros(num), torch.zeros(num)
        with torch.no_grad():
            for i, (img, labels, bg, fg, name) in enumerate(self.test_dataset):
                images = self.transform(img).unsqueeze(0)
                labels = self.t_transform(labels).unsqueeze(0)
                shape = labels.size()[2:]
                images = images.cuda()
                prob_pred = self.net(images, mode='test')
                bg_pred = torch.mean(torch.cat([prob_pred[i+7] for i in self.select], dim=1), dim=1, keepdim=True)
                bg_pred = (bg_pred > 0.5).float()
                prob_pred = torch.mean(torch.cat([prob_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)
                
                prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
                bg_pred = F.interpolate(bg_pred, size=shape, mode='nearest').cpu().data.numpy()
                fork_bg, fork_fg = Bwdist(bg_pred)
                if use_crf:
                    prob_pred = crf(img, prob_pred.numpy(), to_tensor=True)
                if not os.path.exists('{}/visualize_pred{}/'.format(self.visual_save_fold, epoch)):
                    os.mkdir('{}/visualize_pred{}/'.format(self.visual_save_fold, epoch))
                img_save = prob_pred.numpy()
                img_save = img_save.reshape(-1, img_save.shape[2], img_save.shape[3]).transpose(1,2,0) * 255
                cv2.imwrite('{}/visualize_pred{}/{}'.format(self.visual_save_fold, epoch, name), img_save.astype(np.uint8))
                # print('save visualize_pred{}/{} done.'.format(name, epoch))
                if not os.path.exists('{}/visualize_bg{}/'.format(self.visual_save_fold, epoch)):
                    os.mkdir('{}/visualize_bg{}/'.format(self.visual_save_fold, epoch))
                img_save = fork_bg
                img_save = img_save.reshape(-1, img_save.shape[2], img_save.shape[3]).transpose(1,2,0) * 255
                cv2.imwrite('{}/visualize_bg{}/{}'.format(self.visual_save_fold, epoch, name), img_save.astype(np.uint8))
                # print('save visualize_bg{}/{} done.'.format(name, epoch))
                if not os.path.exists('{}/visualize_fg{}/'.format(self.visual_save_fold, epoch)):
                    os.mkdir('{}/visualize_fg{}/'.format(self.visual_save_fold, epoch))
                img_save = fork_fg
                img_save = img_save.reshape(-1, img_save.shape[2], img_save.shape[3]).transpose(1,2,0) * 255
                cv2.imwrite('{}/visualize_fg{}/{}'.format(self.visual_save_fold, epoch, name), img_save.astype(np.uint8))
                # print('save visualize_bg{}/{} done.'.format(name, epoch))
                mae = self.eval_mae(prob_pred, labels)
                if mae == mae:
                    avg_mae += mae
                    img_num += 1.0
                    # prec, recall = self.eval_pr(prob_pred, labels, num)
                    # avg_prec, avg_recall = avg_prec + prec, avg_recall + recall
        avg_mae = avg_mae / img_num
        # avg_mae, avg_prec, avg_recall = avg_mae / img_num, avg_prec / img_num, avg_recall / img_num
        # score = (1 + self.beta ** 2) * avg_prec * avg_recall / (self.beta ** 2 * avg_prec + avg_recall)
        # score[score != score] = 0  # delete the nan
        # print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()))
        print('average mae: %.4f' % (avg_mae))
        # print('average mae: %.4f, max fmeasure: %.4f' % (avg_mae, score.max()), file=self.test_output)
        return avg_mae, 1.0 #score.max()

    # training phase
    def train(self):
        start_epoch = 0
        best_mae = 1.0 if self.config.val else None
        if self.config.resume is not None:
            if not os.path.isfile(self.config.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(self.config.resume)
            start_epoch = checkpoint['epoch']
            if self.config.cuda:
                self.net.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.net.load_state_dict(checkpoint['state_dict'])
            
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            best_mae = checkpoint['best_mae']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.config.resume, checkpoint['epoch']))

        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        
        for epoch in range(start_epoch, self.config.epoch):
            # if str(epoch + 1) in self.step:
            #     self.update_lr()
            loss_epoch = 0
            tbar = tqdm(self.train_loader)
            
            for i, data_batch in enumerate(tbar):
                if (i + 1) > iter_num: break
                self.net.zero_grad()
                x, y, bg, fg= data_batch
                x, y, bg, fg = x.cuda(), y.cuda(), bg.cuda(), fg.cuda()
                y_pred = self.net(x, bg=bg, fg=fg)
                loss = self.loss(y_pred, y)
                loss.backward()
                utils.clip_grad_norm_(self.net.parameters(), self.config.clip_gradient)
                # utils.clip_grad_norm(self.loss.parameters(), self.config.clip_gradient)
                # if (i+1) % self.update == 0 or (i+1) == iter_num:
                self.optimizer.step()
                
                loss_epoch += loss.item()
                self.writer.add_scalar('train/total_loss_iter', loss.item(), epoch * iter_num  + i)
                tbar.set_description('epoch:[%d/%d],loss:[%.4f]' % (
                    epoch, self.config.epoch, loss.item()))
                # print('epoch: [%d/%d], iter: [%d/%d], loss: [%.4f]' % (
                #     epoch, self.config.epoch, i, iter_num, loss.item()))
                if self.config.visdom:
                    error = OrderedDict([('loss:', loss.item())])
                    self.visual.plot_current_errors(epoch, i / iter_num, error)
            self.writer.add_scalar('train/total_loss_epoch', loss_epoch / iter_num, epoch)
            if (epoch + 1) % self.config.epoch_show == 0:
                print('epoch: [%d/%d], epoch_loss: [%.4f]' % (epoch, self.config.epoch, loss_epoch / iter_num),
                      file=self.log_output)
                if self.config.visdom:
                    avg_err = OrderedDict([('avg_loss', loss_epoch / iter_num)])
                    self.visual.plot_current_errors(epoch, i / iter_num, avg_err, 1)
                    y_show = torch.mean(torch.cat([y_pred[i] for i in self.select], dim=1), dim=1, keepdim=True)
                    img = OrderedDict([('origin', x.cpu()[0] * self.std + self.mean), ('label', y.cpu()[0][0]),
                                       ('pred_label', y_show.cpu().data[0][0])])
                    self.visual.plot_current_img(img)
            if self.config.val and (epoch + 1) % self.config.epoch_val == 0:
                mae, fscore = self.test(100, epoch=epoch+1)
                self.writer.add_scalar('test/MAE', mae, epoch)
                self.writer.add_scalar('test/F-Score', fscore, epoch)
                print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae))
                print('--- Best MAE: %.2f, Curr MAE: %.2f ---' % (best_mae, mae), file=self.log_output)
                if best_mae > mae:
                    best_mae = mae
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': self.net.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_mae': mae
                    }, '%s/models/best.pth' % self.config.save_fold)
                    # torch.save(self.net.state_dict(), '%s/models/best.pth' % self.config.save_fold)
            # if (epoch + 1) % self.config.epoch_save == 0:
            #     torch.save(self.net.module.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_fold, epoch + 1))
        torch.save(self.net.module.state_dict(), '%s/models/final.pth' % self.config.save_fold)
