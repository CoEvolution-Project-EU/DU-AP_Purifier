#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import datetime
import os
import time
import copy

# import imp
import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from dataset.kitti.parser import Parser

import torch.optim as optim
from tensorboardX import SummaryWriter as Logger
from common.sync_batchnorm.batchnorm import convert_model
from modules.scheduler.warmupLR import warmupLR
from modules.scheduler.consine import CosineAnnealingWarmUpRestarts

from modules.loss.Lovasz_Softmax import Lovasz_softmax, Lovasz_softmax_PointCloud
from modules.loss.boundary_loss import BoundaryLoss
from modules.utils import AverageMeter, iouEval, save_checkpoint, show_scans_in_training, save_to_txtlog, make_log_img
from modules.denoiser import ResNet_denoiser
from modules.super_resolution import Super_Resolution


import sys
import sys
sys.path.append("/home/mips/Workspace/RangeSeg-main/build")
from image_projection import RangeImageLabeling


class Trainer():
    def __init__(self, ARCH, DATA, datadir, logdir, path=None, point_refine=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.path = path
        self.epoch = 0
        self.point_refine = point_refine
        self.pipeline = self.ARCH["train"]["pipeline"]

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()

        # put logger where it belongs
        self.tb_logger = Logger(self.logdir + "/tb")
        self.info = {"train_update": 0,
                     "train_loss": 0, "train_acc": 0, "train_iou": 0,
                     "valid_loss": 0, "valid_acc": 0, "valid_iou": 0,
                     "best_train_iou": 0, "best_val_iou": 0}

        # get the data
        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA["split"]["train"],
                             valid_sequences=self.DATA["split"]["valid"],
                             test_sequences=None,
                             split='train',
                             labels=self.DATA["labels"],
                             color_map=self.DATA["color_map"],
                             learning_map=self.DATA["learning_map"],
                             learning_map_inv=self.DATA["learning_map_inv"],
                             sensor=self.ARCH["dataset"]["sensor"],
                             max_points=self.ARCH["dataset"]["max_points"],
                             batch_size=self.ARCH["train"]["batch_size"],
                             workers=self.ARCH["train"]["workers"],
                             gt=True,
                             shuffle_train=True)

        self.set_loss_weight()
        self.set_model()
        self.set_gpu_cuda()
        self.set_loss_function(point_refine)
        self.set_optim_scheduler()

        # if need load the pre-trained model from checkpoint
        if self.path is not None:
            self.load_pretrained_model()

    def convert_relu_to_softplus(self, model, act):
        for child_name, child in model.named_children():
            if isinstance(child, nn.LeakyReLU):
                setattr(model, child_name, act)
            else:
                self.convert_relu_to_softplus(child, act)

    def set_model(self):
        with torch.no_grad():
            if self.pipeline == "LENet":
                from modules.network.LENet import ResNet_34
                self.model = ResNet_34(self.parser.get_n_classes(), self.ARCH)

                # #SR model
                self.sr_model = Super_Resolution(ResNet_denoiser(1, channels = [8, 16, 32]), 5)
                weights_path = '/home/mips/Workspace/RangeSeg-main/logs/test_new_pretrained_small_loss_orderSR/2024-11-27-22:48/model_SR.pth'
                state_dict = torch.load(weights_path)
                self.sr_model.load_state_dict(state_dict, strict=False)

            if self.ARCH["train"]["act"] == "Hardswish":
                self.convert_relu_to_softplus(self.model, nn.Hardswish())
            elif self.ARCH["train"]["act"] == "SiLU":
                self.convert_relu_to_softplus(self.model, nn.SiLU())
            elif self.ARCH["train"]["act"] == "GELU":
                self.convert_relu_to_softplus(self.model, nn.GELU())

        save_to_txtlog(self.logdir, 'model.txt', str(self.model))
        pytorch_total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of parameters: ", pytorch_total_params/1000000, "M")
        save_to_txtlog(self.logdir, 'model.txt', "Number of parameters: %.5f M" % (
            pytorch_total_params/1000000))

    def set_loss_weight(self):
        """
            Used to calculate the weights for each class
            weights for loss (and bias)
        """
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        for cl, freq in self.DATA["content"].items():
            # map actual class to xentropy class
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        # ignore the ones necessary to ignore
        for x_cl, w in enumerate(self.loss_w):
            if self.DATA["learning_ignore"][x_cl]:    # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

    def set_loss_function(self, point_refine):
        """
            Used to define the loss function, multiple gpus need to be parallel
            # self.dice = DiceLoss().to(self.device)
            # self.dice = nn.DataParallel(self.dice).cuda()
        """
        # self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
        self.criterion = nn.NLLLoss(
            weight=self.loss_w.double()).to(self.device)
        self.bd = BoundaryLoss().to(self.device)
        if not point_refine:
            self.ls = Lovasz_softmax(ignore=0).to(self.device)
        else:
            self.ls = Lovasz_softmax_PointCloud(ignore=0).to(self.device)

        # loss as dataparallel too (more images in batch)
        if self.n_gpus > 1:
            self.criterion = nn.DataParallel(
                self.criterion).cuda()  # spread in gpus
            self.ls = nn.DataParallel(self.ls).cuda()
            self.bd = nn.DataParallel(self.bd).cuda()

    def set_gpu_cuda(self):
        """
            Used to set gpus and cuda information
        """
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # cudnn.benchmark = True
            # cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()
            self.sr_model.cuda()

    def set_optim_scheduler(self):
        """
            Used to set the optimizer and scheduler
        """
        if self.ARCH["train"]["scheduler"] == "consine":
            length = self.parser.get_train_size()
            dict = self.ARCH["train"]["consine"]
            if self.ARCH["train"]["optimizer"] == "sgd":
                self.optimizer = optim.SGD([{'params': self.model.parameters()}],
                                           lr=dict["min_lr"],
                                           momentum=self.ARCH["train"]["sgd"]["momentum"],
                                           weight_decay=self.ARCH["train"]["sgd"]["w_decay"])
            elif self.ARCH["train"]["optimizer"] == "adam":
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=dict["min_lr"])

            self.scheduler = CosineAnnealingWarmUpRestarts(optimizer=self.optimizer,
                                                           T_0=dict["first_cycle"] * length, T_mult=dict["cycle"],
                                                           eta_max=dict["max_lr"],
                                                           T_up=dict["wup_epochs"]*length, gamma=dict["gamma"])
        elif self.ARCH["train"]["scheduler"] == "warmup":
            steps_per_epoch = self.parser.get_train_size()
            up_steps = int(self.ARCH["train"]["warmup"]
                           ["wup_epochs"] * steps_per_epoch)
            final_decay = self.ARCH["train"]["warmup"]["lr_decay"] ** (
                1 / steps_per_epoch)

            if self.ARCH["train"]["optimizer"] == "sgd":
                self.optimizer = optim.SGD([{'params': self.model.parameters()}],
                                           lr=self.ARCH["train"]["warmup"]["lr"],
                                           momentum=self.ARCH["train"]["sgd"]["momentum"],
                                           weight_decay=self.ARCH["train"]["sgd"]["w_decay"])
            elif self.ARCH["train"]["optimizer"] == "adam":
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=self.ARCH["train"]["warmup"]["lr"])

            self.scheduler = warmupLR(optimizer=self.optimizer,
                                      lr=self.ARCH["train"]["warmup"]["lr"],
                                      warmup_steps=up_steps,
                                      momentum=self.ARCH["train"]["warmup"]["momentum"],
                                      decay=final_decay)

    def load_pretrained_model(self):
        """
            If you want to resume training, reload the model
        """
        torch.nn.Module.dump_patches = True
        checkpoint = self.pipeline + "_train_best"
        w_dict = torch.load(f"{self.path}/{checkpoint}",
                            map_location=lambda storage, loc: storage)
        self.model.load_state_dict(w_dict['state_dict'], strict=True)
        # self.model.load_state_dict({k.replace('module.',''):v for k,v in w_dict['state_dict'].items()})
        # self.optimizer.load_state_dict(w_dict['optimizer'])
        print("load the coarse model of {}".format(checkpoint))

    def calculate_estimate(self, epoch, iter):
        estimate = int((self.data_time_t.avg + self.batch_time_t.avg) *
                       (self.parser.get_train_size() * self.ARCH['train']['max_epochs'] - (
                           iter + 1 + epoch * self.parser.get_train_size()))) + \
            int(self.batch_time_e.avg * self.parser.get_valid_size() * (
                self.ARCH['train']['max_epochs'] - (epoch)))
        return str(datetime.timedelta(seconds=estimate))



    def init_evaluator(self):
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.parser.get_n_classes(),
                                 self.device, self.ignore_class)

    def train(self):
        self.init_evaluator()

        # train for n epochs
        for epoch in range(1):



            acc, iou, loss, rand_img, hetero_l = self.validate(val_loader=self.parser.get_valid_set(),
                                                                   model=self.model,
                                                                   criterion=self.criterion,
                                                                   evaluator=self.evaluator,
                                                                   class_func=self.parser.get_xentropy_class_string,
                                                                   color_fn=self.parser.to_color,
                                                                   save_scans=self.ARCH["train"]["save_scans"])

            self.update_validation_info(epoch, acc, iou, loss, hetero_l)



        print('Finished Evaluation')

        return


    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans=False):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        # model.eval()
        # evaluator.reset()
       # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

    # with torch.no_grad():
        end = time.time()
        for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name,
                _, _, _, _, _, _, _, _, _)\
                in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(non_blocking=True).long()

            if (i==0):
                self.save_range_image(in_vol[0, 0, :,:], in_vol[0, 0, :,:], proj_labels[0,:,:], 200, i)  

            adv_in_vol = torch.zeros_like(in_vol)
            low_res_index = torch.arange(0, 64, 4)
            adv_in_vol[:, :, low_res_index, :] = copy.deepcopy(in_vol[:, :, ::4, :]).cuda()
                
            adv_in_vol = self.generate_the_adversarial_points(adv_in_vol, proj_labels, model, criterion, 0.05)
            adv_in_vol = adv_in_vol.detach()

            # adv_in_vol = (in_vol.to(torch.int16)).to(torch.float32)

            # Save range images at specified intervals
            if (i==0):
                self.save_range_image(adv_in_vol[0, 0, :,:], in_vol[0, 0, :,:], proj_labels[0,:,:], 100, i)  

            model.eval()
            # compute output
            if self.ARCH["train"]["aux_loss"]["use"]:
                output, _, _, _ = model(adv_in_vol)
            else:
                output, _ = model(adv_in_vol)
            log_out = torch.log(output.clamp(min=1e-8))

            # wce = criterion(log_out, proj_labels)
            jacc = self.ls(output, proj_labels)
            wce = criterion(log_out.double(), proj_labels).float()
            loss = wce + jacc

            # measure accuracy and record loss
            argmax = output.argmax(dim=1)
            evaluator.addBatch(argmax, proj_labels)

            losses.update(loss.mean().item(), in_vol.size(0))
            jaccs.update(jacc.mean().item(), in_vol.size(0))
            wces.update(wce.mean().item(), in_vol.size(0))

            if save_scans:
                # get the first scan in batch and project points
                mask_np = proj_mask[0].cpu().numpy()
                depth_np = adv_in_vol[0][0].cpu().numpy()
                pred_np = argmax[0].cpu().numpy()
                gt_np = proj_labels[0].cpu().numpy()
                out = make_log_img(depth_np, mask_np,
                                    pred_np, gt_np, color_fn)
                rand_imgs.append(out)

            # measure elapsed time
            self.batch_time_e.update(time.time() - end)
            end = time.time()

        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), in_vol.size(0))
        iou.update(jaccard.item(), in_vol.size(0))

        str_line = ("*" * 80 + '\n'
                    'Validation set:\n'
                    'Time avg per batch {batch_time.avg:.3f}\n'
                    'Loss avg {loss.avg:.4f}\n'
                    'Jaccard avg {jac.avg:.4f}\n'
                    'WCE avg {wces.avg:.4f}\n'
                    'Acc avg {acc.avg:.6f}\n'
                    'IoU avg {iou.avg:.6f}').format(
                        batch_time=self.batch_time_e, loss=losses,
                        jac=jaccs, wces=wces, acc=acc, iou=iou)
        print(str_line)
        save_to_txtlog(self.logdir, 'log.txt', str_line)

        # print also classwise
        for i, jacc in enumerate(class_jaccard):
            self.info["valid_classes/" + class_func(i)] = jacc
            str_line = 'IoU class {i:} [{class_str:}] = {jacc:.6f}'.format(
                i=i, class_str=class_func(i), jacc=jacc)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)
        str_line = '*' * 80
        print(str_line)
        save_to_txtlog(self.logdir, 'log.txt', str_line)

        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg

    def update_training_info(self, epoch, acc, iou, loss, update_mean, hetero_l):
        # update info
        self.info["train_update"] = update_mean
        self.info["train_loss"] = loss
        self.info["train_acc"] = acc
        self.info["train_iou"] = iou
        self.info["train_hetero"] = hetero_l

        # remember best iou and save checkpoint
        state = {'epoch': epoch,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'info': self.info,
                 'scheduler': self.scheduler.state_dict()}
        save_checkpoint(state, self.logdir, self.pipeline, suffix="")

        if self.info['train_iou'] > self.info['best_train_iou']:
            print("Best mean iou in training set so far, save model!")
            self.info['best_train_iou'] = self.info['train_iou']
            state = {'epoch': epoch,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()}
            save_checkpoint(state, self.logdir, self.pipeline,
                            suffix="_train_best")

    def update_validation_info(self, epoch, acc, iou, loss, hetero_l):
        # update info
        self.info["valid_loss"] = loss
        self.info["valid_acc"] = acc
        self.info["valid_iou"] = iou
        self.info['valid_heteros'] = hetero_l

        # remember best iou and save checkpoint
        if self.info['valid_iou'] > self.info['best_val_iou']:
            str_line = (
                "Best mean iou in validation so far, save model!\n" + "*" * 80)
            print(str_line)
            save_to_txtlog(self.logdir, 'log.txt', str_line)
            self.info['best_val_iou'] = self.info['valid_iou']

            # save the weights!
            state = {'epoch': epoch,
                     'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'info': self.info,
                     'scheduler': self.scheduler.state_dict()}
            save_checkpoint(state, self.logdir, self.pipeline,
                            suffix="_valid_best")

            str_line = ("*" * 80 + '\n'
                        'Validation set:\n'
                        'epoch {epoch}:\n'
                        'Loss avg {loss}\n'
                        'Acc avg {acc}\n'
                        'IoU avg {iou}').format(
                            epoch=epoch, loss=loss,
                            acc=acc, iou=iou)
            save_to_txtlog(self.logdir, 'validate.txt', str_line)


    def generate_the_adversarial_points(self, in_vol, proj_labels, model, criterion, eps):
        """
        Args:
            in_vol (torch.Tensor): Input range images (batch_size, 1, 64, 1024).
            proj_labels (torch.Tensor): Ground truth segmentation masks (batch_size, 64, 1024).
            model (torch.nn.Module): The segmentation network.
            epsilon (float): Perturbation magnitude.
            criterion (function): Loss function (e.g., cross-entropy).

        Returns:
            adv_in_vol (torch.Tensor): Adversarially perturbed input range images.
        """

        iter_eps = eps/30
        nb_iter = 15            #/FGSM this is one
        norm = 2 # np.inf 2
        decay_factor = 1
        clip_min = None
        clip_max = None

        key_origin = in_vol.clone()
        key_origin.requires_grad = False # important
        
        model.train()
        in_vol.requires_grad = True
        g = torch.zeros_like(key_origin).to(key_origin.device)        

        for i in range(nb_iter):
            
            # Forward pass
            if self.ARCH["train"]["aux_loss"]["use"]:
                output, _, _, _ = model(in_vol)
            else:
                output, _ = model(in_vol)


            # Compute the loss
            log_out = torch.log(output.clamp(min=1e-8))
            jacc = self.ls(output, proj_labels)
            wce = criterion(log_out.double(), proj_labels).float()
            loss = wce + jacc

            # Backward pass to compute gradient w.r.t. input
            model.zero_grad()
            in_vol.retain_grad()
            loss.backward()
            # Extract the gradient of the loss w.r.t. the input
            grad = in_vol.grad.data
            perturbation = self.clip_eta(grad, iter_eps, norm)

            in_vol.requires_grad = False

            perturbation = in_vol + perturbation - key_origin
            perturbation = self.clip_eta(perturbation, eps, norm)

            # Apply the perturbation to the input
            in_vol =  key_origin + perturbation


            in_vol = in_vol.detach()
            # Detach gradients from the perturbed input
            in_vol.requires_grad = True

        return in_vol

    # def generate_the_adversarial_points(self, in_vol, proj_labels, model, criterion, epsilon):
    #     """
    #     Generate adversarial points using FGSM for segmentation.

    #     Args:
    #         in_vol (torch.Tensor): Input range images (batch_size, 1, 64, 1024).
    #         proj_labels (torch.Tensor): Ground truth segmentation masks (batch_size, 64, 1024).
    #         model (torch.nn.Module): The segmentation network.
    #         epsilon (float): Perturbation magnitude.
    #         criterion (function): Loss function (e.g., cross-entropy).

    #     Returns:
    #         adv_in_vol (torch.Tensor): Adversarially perturbed input range images.
    #     """
    #     model.train()  
     
        
    #     in_vol.requires_grad = True
        


    #     # Forward pass
    #     if self.ARCH["train"]["aux_loss"]["use"]:
    #         output, _, _, _ = model(in_vol)
    #     else:
    #         output, _ = model(in_vol)


    #     # Compute the loss
    #     log_out = torch.log(output.clamp(min=1e-8))
    #     jacc = self.ls(output, proj_labels)
    #     wce = criterion(log_out.double(), proj_labels).float()
    #     loss = wce + jacc

    #     # Backward pass to compute gradient w.r.t. input
    #     model.zero_grad()
    #     loss.backward()  # This computes gradients for in_vol

    #     # Extract the gradient of the loss w.r.t. the input
    #     grad = in_vol.grad.data

    #     # Generate perturbation using the sign of the gradient
    #     perturbation = epsilon * torch.sign(grad)

    #     # Apply the perturbation to the input
    #     adv_in_vol = in_vol + perturbation

    #     # Clip the adversarial input to ensure valid range (e.g., [0, 1])
    #     # adv_in_vol = torch.clamp(adv_in_vol, 0, 1)

    #     # Detach gradients from the perturbed input
    #     adv_in_vol = adv_in_vol.detach()

    #     return adv_in_vol







    def save_range_image(self, image, mask, labels, epoch, batch, output_dir="range_images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = f"{output_dir}/range_image_epoch_{epoch}_batch_{batch}"
        mfilename = f"{output_dir}/mask_image_epoch_{epoch}_batch_{batch}"
        lfilename = f"{output_dir}/labels_image_epoch_{epoch}_batch_{batch}"

        image_np = image.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        np.save(filename, image_np)
        np.save(mfilename, mask_np)
        # np.save(lfilename, labels_np)






    
    def clip_eta(self, grad, eps, norm=np.inf):
        """
        Solves for the optimal input to a linear function under a norm constraint.
        Optimal_perturbation = argmax_{eta, ||eta||_{norm} < eps} dot(eta, grad)
        :param grad: Tensor, shape (N, d_1, ...). Batch of gradients
        :param eps: float. Scalar specifying size of constraint region
        :param norm: np.inf, 1, or 2. Order of norm constraint.
        :returns: Tensor, shape (N, d_1, ...). Optimal perturbation
        """
        grad_shape = grad.shape
        grad_shape_len = len(grad.shape)
        if grad_shape_len == 3:
            grad = grad.view(-1, 3)

        red_ind = list(range(1, len(grad.size())))
        avoid_zero_div = torch.tensor(1e-36, dtype=grad.dtype, device=grad.device)
        if norm == np.inf:
            # Take sign of gradient
            optimal_perturbation = torch.sign(grad)
        elif norm == 1:
            abs_grad = torch.abs(grad)
            sign = torch.sign(grad)
            red_ind = list(range(1, len(grad.size())))
            ori_shape = [1] * len(grad.size())
            ori_shape[0] = grad.size(0)

            max_abs_grad, _ = torch.max(abs_grad.view(grad.size(0), -1), 1)
            max_mask = abs_grad.eq(max_abs_grad.view(ori_shape)).to(torch.float)
            num_ties = max_mask
            for red_scalar in red_ind:
                num_ties = torch.sum(num_ties, red_scalar, keepdim=True)
            optimal_perturbation = sign * max_mask / num_ties
            # TODO integrate below to a test file
            # check that the optimal perturbations have been correctly computed
            opt_pert_norm = optimal_perturbation.abs().sum(dim=red_ind)
            assert torch.all(opt_pert_norm == torch.ones_like(opt_pert_norm))
        elif norm == 2:
            square = torch.sum(grad ** 2, red_ind, keepdim=True)
            optimal_perturbation = grad / torch.max(torch.sqrt(square), avoid_zero_div)
            # TODO integrate below to a test file
            # check that the optimal perturbations have been correctly computed
            opt_pert_norm = (
                optimal_perturbation.pow(2).sum(dim=red_ind, keepdim=True).sqrt()
            )
            one_mask = (square <= avoid_zero_div).to(torch.float) * opt_pert_norm + (
                square > avoid_zero_div
            ).to(torch.float)
            assert torch.allclose(opt_pert_norm, one_mask, rtol=1e-05, atol=1e-08)
        else:
            raise NotImplementedError(
                "Only L-inf, L1 and L2 norms are " "currently implemented."
            )

        # Scale perturbation to be the solution for the norm=eps rather than
        # norm=1 problem
        scaled_perturbation = eps * optimal_perturbation
        if grad_shape_len == 3:
            scaled_perturbation = scaled_perturbation.view(grad_shape)
        return scaled_perturbation