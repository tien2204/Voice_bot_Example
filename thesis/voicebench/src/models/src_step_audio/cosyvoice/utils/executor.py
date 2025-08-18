














import logging
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist

from cosyvoice.utils.train_utils import (
    update_parameter_and_lr,
    log_per_step,
    log_per_save,
    batch_forward,
    batch_backward,
    save_model,
    cosyvoice_join,
)


class Executor:

    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.rank = int(os.environ.get("RANK", 0))
        self.device = torch.device("cuda:{}".format(self.rank))

    def train_one_epoc(
        self,
        model,
        optimizer,
        scheduler,
        train_data_loader,
        cv_data_loader,
        writer,
        info_dict,
        group_join,
    ):
        """Train one epoch"""

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch {} TRAIN info lr {} rank {}".format(self.epoch, lr, self.rank)
        )
        logging.info(
            "using accumulate grad, new batch size is {} times"
            " larger than before".format(info_dict["accum_grad"])
        )
        
        
        
        model.train()
        model_context = (
            model.join if info_dict["train_engine"] == "torch_ddp" else nullcontext
        )
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                
                
                
                if (
                    info_dict["train_engine"] == "torch_ddp"
                    and (batch_idx + 1) % info_dict["accum_grad"] != 0
                ):
                    context = model.no_sync
                
                
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, info_dict)
                    info_dict = batch_backward(model, info_dict)

                info_dict = update_parameter_and_lr(
                    model, optimizer, scheduler, info_dict
                )
                log_per_step(writer, info_dict)
                
                if (
                    info_dict["save_per_step"] > 0
                    and (self.step + 1) % info_dict["save_per_step"] == 0
                    and (batch_idx + 1) % info_dict["accum_grad"] == 0
                ):
                    dist.barrier()
                    self.cv(
                        model, cv_data_loader, writer, info_dict, on_batch_end=False
                    )
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        """Cross validation on"""
        logging.info(
            "Epoch {} Step {} on_batch_end {} CV rank {}".format(
                self.epoch, self.step + 1, on_batch_end, self.rank
            )
        )
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            info_dict = batch_forward(model, batch_dict, info_dict)

            for k, v in info_dict["loss_dict"].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict["loss_dict"] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = (
            "epoch_{}_whole".format(self.epoch)
            if on_batch_end
            else "epoch_{}_step_{}".format(self.epoch, self.step + 1)
        )
        save_model(model, model_name, info_dict)
