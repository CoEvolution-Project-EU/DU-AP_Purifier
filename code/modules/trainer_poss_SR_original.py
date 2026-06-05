#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import datetime
import os
import copy
import time
import imp
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from modules.trainer_for_attack import Trainer
from dataset.poss.parser import Parser
import __init__ as booger


from modules.utils import AverageMeter, iouEval, save_checkpoint, show_scans_in_training, save_to_txtlog, make_log_img
from modules.denoiser import ResNet_denoiser
from modules.model_based_denoiser import DU_denoiser



# import sys
# sys.path.append("/home/mips/Workspace/RangeSeg-main/build")
# from image_projection import RangeImageLabeling


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print( "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

class TrainerPoss(Trainer):
    def __init__(self, ARCH, DATA, datadir, logdir, path=None):
        super(TrainerPoss, self).__init__(ARCH, DATA,
                                          datadir, logdir, path, point_refine=False)

        # get the data
        self.parser = Parser(root=self.datadir,
                             train_sequences=self.DATA["split"]["train"],
                             valid_sequences=self.DATA["split"]["valid"],
                             test_sequences=None,
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

    def train_epoch(self, train_loader, model, criterion, optimizer,
                    epoch, evaluator, scheduler, color_fn, report=10,
                    show_scans=False):

        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()
        bd = AverageMeter()
        criterion_sr = nn.L1Loss()
        # sobel_loss = SobelLoss()
        # empty the cache to train now
        # if self.gpu:
        #     torch.cuda.empty_cache()

        # switch to train mode
        model.eval()
        self.sr_model.train()

        end = time.time()
        for i, (in_vol, proj_labels, _, _, path_seq, path_name, _, _, _, _, _) in enumerate(train_loader):
            # measure data loading time
            self.data_time_t.update(time.time() - end)

            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
                #proj_mask = proj_mask.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()
                
            # low_res_input = in_vol[:, :, ::4, :]

            adv_in_vol = self.generate_the_adversarial_points(in_vol, proj_labels, model, criterion, 1)
            # print("-------------------------yes--------------------")
            adv_in_vol = adv_in_vol.detach() 


            sr_output = self.sr_model(adv_in_vol) 

            loss_sr = criterion_sr(sr_output, in_vol)
            
            
            # masked_loss = torch.abs(sr_output - in_vol) * w
            # loss_sr = masked_loss.sum() / w.sum()         
            



            if self.ARCH["train"]["aux_loss"]["use"]:
                
                [output, z2, z4, z8] = model(sr_output)
                
                lamda = self.ARCH["train"]["aux_loss"]["lamda"]
                bdlosss = self.bd(output, proj_labels.long()) + lamda[0]*self.bd(z2, proj_labels.long(
                )) + lamda[1]*self.bd(z4, proj_labels.long()) + lamda[2]*self.bd(z8, proj_labels.long())
                loss_m0 = criterion(torch.log(output.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(output, proj_labels.long())
                loss_m2 = criterion(torch.log(z2.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z2, proj_labels.long())
                loss_m4 = criterion(torch.log(z4.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z4, proj_labels.long())
                loss_m8 = criterion(torch.log(z8.clamp(
                    min=1e-8)).double(), proj_labels).float() + 1.5 * self.ls(z8, proj_labels.long())
                loss_m = loss_m0 + lamda[0]*loss_m2 + \
                    lamda[1]*loss_m4 + lamda[2]*loss_m8 + bdlosss + 10*loss_sr
            else:
                output, _ = model(sr_output)
                bdlosss = self.bd(output, proj_labels.long())
                # loss_m = criterion(torch.log(output.clamp(
                #     min=1e-8)).double(), proj_labels).float() + self.ls(output, proj_labels.long()) + bdlosss + 10*loss_sr
                loss_m = loss_sr

            # # Save range images at specified intervals
            # if (epoch %1 == 0 and i==0):
            #     save_range_image(sr_output[0, 0, :,:], proj_labels[0, :,:], output_mask[0, :,:], output_mask[0, :,:],  in_vol[0, 0, :, :],  output.argmax(dim=1) [0, :, :], epoch, i)  

            optimizer.zero_grad()
            if self.n_gpus > 1:
                idx = torch.ones(self.n_gpus).cuda()
                loss_m.backward(idx)
                # nn.utils.clip_grad.clip_grad_norm_(
                #    self.sr_model.parameters(), max_norm=1, norm_type=2)
                # nn.utils.clip_grad.clip_grad_norm_(
                    # self.sr_model.parameters(), max_norm=1, norm_type=2)
            else:
                loss_m.backward()
                # nn.utils.clip_grad.clip_grad_norm_(
                    # self.sr_model.parameters(), max_norm=1, norm_type=2)
                # nn.utils.clip_grad.clip_grad_norm_(
                    # self.sr_model.parameters(), max_norm=1, norm_type=2)
            
            
            
            
            # for name, param in self.sr_model.named_parameters():
            #     if name.startswith("attention") and param.grad is not None:
            #     # if param.grad is not None:
            #         print(f"Layer: {name} | Grad Mean: {param.grad.mean().item():.6f} | Grad Std: {param.grad.std().item():.6f}")

                            
                
            
            
            optimizer.step()
            # measure accuracy and record loss
            loss = loss_m.mean()
            bd_loss = bdlosss.mean()
            with torch.no_grad():
                evaluator.reset()
                argmax = output.argmax(dim=1)
                evaluator.addBatch(argmax, proj_labels)
                accuracy = evaluator.getacc()
                jaccard, class_jaccard = evaluator.getIoU()

            losses.update(loss.item(), in_vol.size(0))
            acc.update(accuracy.item(), in_vol.size(0))
            iou.update(jaccard.item(), in_vol.size(0))
            bd.update(bd_loss.item(), in_vol.size(0))

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            # get gradient updates and weights, so I can print the relationship of
            # their norms
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(
                            value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-max(lr, 1e-10)
                                                * value.grad.cpu().numpy().reshape((-1)))
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)  # over the epoch

            if show_scans:
                # get the first scan in batch and project points
                depth_np = in_vol[0][0].cpu().numpy()
                pred_np = argmax[0].cpu().numpy()
                gt_np = proj_labels[0].cpu().numpy()
                out = self.make_log_img(depth_np, pred_np, gt_np, color_fn)

                directory = os.path.join(self.log, "train-predictions")
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, out)

            if i % self.ARCH["train"]["report_batch"] == 0:
                str_line = ('Lr: {lr:.3e} | '
                            'Update: {umean:.3e} mean,{ustd:.3e} std | '
                            'Epoch: [{0}][{1}/{2}] | '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
                            'Bd {bd.val:.4f} ({bd.avg:.4f}) | '
                            'acc {acc.val:.3f} ({acc.avg:.3f}) | '
                            'IoU {iou.val:.3f} ({iou.avg:.3f}) | [{estim}]').format(
                    epoch, i, len(train_loader), batch_time=self.batch_time_t,
                    data_time=self.data_time_t, loss=losses, bd=bd, acc=acc, iou=iou, lr=lr,
                    umean=update_mean, ustd=update_std, estim=self.calculate_estimate(epoch, i))
                print(str_line)
                save_to_txtlog(self.logdir, 'log.txt', str_line)

            # step scheduler
            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg, hetero_l.avg
    
    def save_range_image(self, image, mode, batch, index, output_dir="range_images"):
        folder = os.path.join(output_dir, mode)
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"range_image_{mode}_batch_{batch}_idx_{index}.npy")
        np.save(filename, image.detach().cpu().numpy())

    def validate(self, val_loader, model, criterion, evaluator, class_func, color_fn, save_scans=False):
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        # switch to evaluate mode
        model.eval()
        self.sr_model.train()
        evaluator.reset()

        # empty the cache to infer in high res
        # if self.gpu:
        #     torch.cuda.empty_cache()
        # range_image_labeling = RangeImageLabeling()

        # with torch.no_grad():
        end = time.time()
        for i, (in_vol, proj_labels, _, _, path_seq, path_name, _, _, _, _, _)\
                in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):
            if not self.multi_gpu and self.gpu:
                in_vol = in_vol.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda(non_blocking=True).long()

            # low_res_input = in_vol[:, :, ::4, :] 
            adv_in_vol = self.generate_the_adversarial_points(in_vol, proj_labels, model, criterion, 1)
            adv_in_vol = adv_in_vol.detach()

            #STORE THE CLEAN AND THE ADVERSARIAL ATTACKS
            # for idx in range(in_vol.size(0)):  # batch loop
            #     self.save_range_image(in_vol[idx, 0], mode="clean", batch=i, index=idx)
            #     self.save_range_image(adv_in_vol[idx, 0], mode="adv", batch=i, index=idx)

            sr_output = self.sr_model(adv_in_vol) 

            #STORE THE CLEAN AND THE ADVERSARIAL ATTACKS
            # for idx in range(in_vol.size(0)):  # batch loop
            #     self.save_range_image(sr_output[idx, 0], mode="purified", batch=i, index=idx)


            # compute output
            if self.ARCH["train"]["aux_loss"]["use"]:
                output, _, _, _ = model(sr_output)
            else:
                output, _ = model(sr_output)
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
                depth_np = in_vol[0][0].cpu().numpy()
                pred_np = argmax[0].cpu().numpy()
                gt_np = proj_labels[0].cpu().numpy()
                out = Trainer.make_log_img(depth_np,
                                            pred_np,
                                            gt_np,
                                            color_fn)

                directory = os.path.join(self.log, "valid-predictions")
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, out)

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

    def make_log_img(depth, pred, gt, color_fn):
        # input should be [depth, pred, gt]
        # make range image (normalized to 0,1 for saving)
        depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        out_img = cv2.applyColorMap(
            depth, Trainer.get_mpl_colormap('viridis'))
        # make label prediction
        pred_color = color_fn(pred.astype(np.int32))
        out_img = np.concatenate([out_img, pred_color], axis=0)
        # make label gt
        gt_color = color_fn(gt)
        out_img = np.concatenate([out_img, gt_color], axis=0)
        return (out_img).astype(np.uint8)
    


# def save_range_image(image_recon, gt_labels, atention_mask, seg_16, in_vol, seg_64_sr, epoch, batch, output_dir="range_images"):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
    
#     # Define unique filenames for each output file.
#     range_image_filename = os.path.join(output_dir, f"range_image_epoch_{epoch}_batch_{batch}.npy")
#     attnetion_mask_filename = os.path.join(output_dir, f"attnetion_mask_epoch_{epoch}_batch_{batch}.npy")
#     lables_gt_filename = os.path.join(output_dir, f"labels_gt_epoch_{epoch}_batch_{batch}.npy")
#     seg_16_filename = os.path.join(output_dir, f"seg_16_epoch_{epoch}_batch_{batch}.npy")
#     in_vol_filename = os.path.join(output_dir, f"in_vol_epoch_{epoch}_batch_{batch}.npy")
#     seg_64_sr_filename = os.path.join(output_dir, f"seg_64_sr_epoch_{epoch}_batch_{batch}.npy")


#     image_recon_np = image_recon.detach().cpu().numpy()
#     gt_labels_np = gt_labels.detach().cpu().numpy()
#     atention_mask_np = atention_mask.detach().cpu().numpy()
#     seg_16_np = seg_16.detach().cpu().numpy()
#     in_vol_np = in_vol.detach().cpu().numpy()
#     seg_64_sr_np = seg_64_sr.detach().cpu().numpy()




#     np.save(range_image_filename, image_recon_np)
#     np.save(attnetion_mask_filename, atention_mask_np)
#     np.save(lables_gt_filename, gt_labels_np)
#     np.save(seg_16_filename, seg_16_np)
#     np.save(in_vol_filename, in_vol_np)
#     np.save(seg_64_sr_filename, seg_64_sr_np)








def projection_to_pointcloud_batch(projection, fov_up = 3, fov_down = -25):
    # Assuming projection is a PyTorch tensor of shape [batch_size, 1, num_lasers, img_length]
    batch_size, _, proj_H, proj_W = projection.shape


    fov_up_rad = (fov_up / 180) * np.pi
    fov_down_rad = (fov_down / 180) * np.pi
    fov_rad = abs(fov_up_rad) + abs(fov_down_rad)

    # Compute the yaw and pitch values for each pixel in the range image
    ys, xs = torch.meshgrid(torch.arange(proj_H), torch.arange(proj_W), indexing='ij')
    ys, xs = ys.to(projection.device), xs.to(projection.device)
    yaw = - (xs * 2.0 * np.pi / proj_W - np.pi)
    pitch = ((proj_H - 1 - ys) * fov_rad / proj_H + fov_down_rad)

    # Expand yaw and pitch to match batch size
    yaw = yaw.unsqueeze(0).expand(batch_size, -1, -1)
    pitch = pitch.unsqueeze(0).expand(batch_size, -1, -1)

    # Convert range and angles to cartesian coordinates
    proj_range = projection.squeeze(1)  # Remove channel dimension
    x = proj_range * torch.cos(pitch) * torch.cos(yaw)
    y = proj_range * torch.cos(pitch) * torch.sin(yaw)
    z = proj_range * torch.sin(pitch)

    # Stack them together and reshape
    point_cloud = torch.stack((x, y, z), dim=-1)
    point_cloud = point_cloud.view(batch_size, -1, 3)  # Reshape to have a list of points per batch

    return point_cloud 


