# # =========================
# # pl_module.py  (FSCSModule)
# # =========================
# import abc
# import os
# import numpy as np
# import torch
# import pytorch_lightning as pl
# from PIL import Image

# from common import utils
# from common.evaluation import AverageMeter


# class FSCSModule(pl.LightningModule, metaclass=abc.ABCMeta):
#     """
#     LIDC setup:
#     - Classification: 5-class on query
#       batch["query_class"]: [B] 0..4
#       output_cls: [B,5]
#     - Segmentation: few-shot binary
#       output_masks: [B,way,2,H,W]

#     Important for pseudo mode:
#       - model must set self._last_query_pmask_all = [B,way,H,W] probabilities.
#       - Here we binarize that pseudo mask with an adaptive threshold for metrics/visualization.
#       - Training loss is computed inside model.compute_objective (it may use soft p, not this binary mask).
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.way = args.way
#         self.sup = args.sup

#         self.learner = None

#         # meters
#         self.avg_meter = {"trn": None, "val": None, "test": None}

#         # classification counters
#         self._cls_correct = {"trn": 0.0, "val": 0.0, "test": 0.0}
#         self._cls_total = {"trn": 0.0, "val": 0.0, "test": 0.0}

#         # loss counters
#         self._loss_sum = {"trn": 0.0, "val": 0.0, "test": 0.0}
#         self._loss_n = {"trn": 0.0, "val": 0.0, "test": 0.0}

#     # -------------------------
#     # abstract
#     # -------------------------
#     @abc.abstractmethod
#     def forward(self, batch):
#         pass

#     @abc.abstractmethod
#     def train_mode(self):
#         pass

#     @abc.abstractmethod
#     def configure_optimizers(self):
#         pass

#     @abc.abstractmethod
#     def predict_cls_seg(self, batch, nshot):
#         pass

#     @abc.abstractmethod
#     def compute_objective(self, output_cls, output_masks, gt_cls, gt_mask, gt_way, lambda_cls=1.0):
#         pass

#     # -------------------------
#     # helpers
#     # -------------------------
#     def predict_cls(self, output_cls):
#         return torch.argmax(output_cls, dim=1)  # [B]

#     def map_class_to_way(self, cls0to4, support_classes_1to5):
#         target = cls0to4.long() + 1  # 1..5
#         return (support_classes_1to5.long() == target.view(-1, 1)).long().argmax(dim=1)

#     def predict_mask_from_way(self, output_masks, way_idx):
#         with torch.no_grad():
#             B, way, C, H, W = output_masks.shape
#             idx = way_idx.view(B, 1, 1, 1, 1).expand(-1, 1, C, H, W)
#             seg_logits = output_masks.gather(1, idx).squeeze(1)
#             return seg_logits.argmax(dim=1).long()

#     # -------------------------
#     # VIS helpers
#     # -------------------------
#     @staticmethod
#     def _tensor_img_to_u8(img_chw: torch.Tensor) -> np.ndarray:
#         """[3,H,W] -> uint8 [H,W,3] via min-max."""
#         x = img_chw.detach().cpu().float()
#         x = x.permute(1, 2, 0).numpy()
#         x = (x - x.min()) / (x.max() - x.min() + 1e-8)
#         return (x * 255.0).astype(np.uint8)

#     @staticmethod
#     def _mask_to_u8(mask_hw: torch.Tensor) -> np.ndarray:
#         """[H,W] 0/1 or float -> uint8 [H,W] 0/255 using >0.5."""
#         m = mask_hw.detach().cpu().float()
#         m = (m > 0.5).to(torch.uint8).numpy()
#         return (m * 255).astype(np.uint8)

#     @staticmethod
#     def _prob_to_u8(prob_hw: torch.Tensor) -> np.ndarray:
#         """[H,W] float in [0,1] -> uint8 grayscale [H,W]."""
#         p = prob_hw.detach().cpu().float().clamp(0, 1).numpy()
#         return (p * 255.0).astype(np.uint8)

#     @staticmethod
#     def _overlay_red(img_u8: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
#         """Overlay mask on image in RED. img_u8: HWC, mask_u8: HW (0/255)."""
#         out = img_u8.astype(np.float32).copy()
#         m = (mask_u8 > 0)
#         if m.any():
#             red = np.zeros_like(out)
#             red[..., 0] = 255
#             out[m] = (1 - alpha) * out[m] + alpha * red[m]
#         return np.clip(out, 0, 255).astype(np.uint8)

#     def _save_episode_vis_onefile(
#         self,
#         batch,
#         split: str,
#         batch_idx: int,
#         gt_way: torch.Tensor,     # [B]
#         pred_seg: torch.Tensor,   # [B,H,W]
#         gt_mask_bin: torch.Tensor,  # [B,H,W] binary (query GT or binarized pseudo)
#         pseudo_prob: torch.Tensor = None,  # [B,H,W] optional
#         max_save: int = 50,
#     ):
#         """
#         Saves ONE PNG per batch_idx with panels:
#           [support overlay] | [query+GT overlay] | [query+PRED overlay] | (optional) [pseudo prob map]
#         """
#         if not getattr(self.args, "vis", False):
#             return
#         if split not in ["val", "test"]:
#             return
#         if batch_idx >= max_save:
#             return
#         if hasattr(self, "trainer") and (not self.trainer.is_global_zero):
#             return

#         out_dir = os.path.join(self.args.logpath, "vis", split)
#         os.makedirs(out_dir, exist_ok=True)

#         b = 0
#         q_img_u8 = self._tensor_img_to_u8(batch["query_img"][b])
#         q_gt_u8 = self._mask_to_u8(gt_mask_bin[b])
#         q_pr_u8 = self._mask_to_u8(pred_seg[b])

#         q_gt_overlay = self._overlay_red(q_img_u8, q_gt_u8)
#         q_pr_overlay = self._overlay_red(q_img_u8, q_pr_u8)

#         way_idx = int(gt_way[b].item())
#         shot_idx = 0
#         s_img_u8 = self._tensor_img_to_u8(batch["support_imgs"][b, way_idx, shot_idx])
#         s_m_u8 = self._mask_to_u8(batch["support_masks"][b, way_idx, shot_idx])
#         s_overlay = self._overlay_red(s_img_u8, s_m_u8)

#         panels = [s_overlay, q_gt_overlay, q_pr_overlay]

#         if pseudo_prob is not None:
#             p_u8 = self._prob_to_u8(pseudo_prob[b])
#             p_rgb = np.stack([p_u8, p_u8, p_u8], axis=-1)
#             panels.append(p_rgb)

#         panel = np.concatenate(panels, axis=1)

#         prefix = f"ep{int(self.current_epoch):03d}_b{int(batch_idx):04d}"
#         save_path = os.path.join(out_dir, f"{prefix}_SUP_QGT_QPR{'_P' if pseudo_prob is not None else ''}.png")
#         Image.fromarray(panel).save(save_path)
#         print(f"[VIS] saved: {save_path}")

#     # -------------------------
#     # epoch starts
#     # -------------------------
#     def on_train_epoch_start(self):
#         utils.fix_randseed(None)
#         if getattr(self.trainer, "rerun", False):
#             self.trainer.optimizers[0].param_groups[0]["capturable"] = True

#         self.avg_meter["trn"] = AverageMeter(self.trainer.train_dataloader.dataset.datasets, self.args.way)

#         self._cls_correct["trn"] = 0.0
#         self._cls_total["trn"] = 0.0
#         self._loss_sum["trn"] = 0.0
#         self._loss_n["trn"] = 0.0

#         self.train_mode()

#     def on_validation_epoch_start(self):
#         self._shared_eval_epoch_start(self.trainer.val_dataloaders[0].dataset, split="val")

#     def on_test_epoch_start(self):
#         test_ds = self.trainer.test_dataloaders[0].dataset
#         self._shared_eval_epoch_start(test_ds, split="test")

#     def _shared_eval_epoch_start(self, dataset, split: str):
#         utils.fix_randseed(0)
#         self.avg_meter[split] = AverageMeter(dataset, self.args.way)

#         self._cls_correct[split] = 0.0
#         self._cls_total[split] = 0.0
#         self._loss_sum[split] = 0.0
#         self._loss_n[split] = 0.0

#         self.eval()

#     # -------------------------
#     # steps
#     # -------------------------
#     def training_step(self, batch, batch_idx):
#         return self._shared_step(batch, batch_idx, "trn")

#     def validation_step(self, batch, batch_idx):
#         self._shared_step(batch, batch_idx, "val")

#     def test_step(self, batch, batch_idx):
#         self._shared_step(batch, batch_idx, "test")

#     def _shared_step(self, batch, batch_idx, split):
#         output_cls, output_masks = self.forward(batch)  # cls [B,5], masks [B,way,2,H,W]

#         if "query_class" not in batch:
#             raise KeyError("batch['query_class'] missing (expected 0..4).")
#         if "support_classes" not in batch:
#             raise KeyError("batch['support_classes'] missing (expected [B,way] 1..5).")

#         gt_cls = batch["query_class"].long()
#         support_classes = batch["support_classes"].long()

#         pred_cls = self.predict_cls(output_cls)
#         pred_way = self.map_class_to_way(pred_cls, support_classes)
#         gt_way = self.map_class_to_way(gt_cls, support_classes)

#         pseudo_prob_gtway = None  # [B,H,W] for visualization if pseudo

#         # --------- choose GT seg mask for METRICS/VIS ----------
#         # NOTE: training loss can ignore this and use soft pmask inside compute_objective.
#         if self.sup == "pseudo":
#             if not hasattr(self, "_last_query_pmask_all") or self._last_query_pmask_all is None:
#                 raise RuntimeError("sup='pseudo' but model._last_query_pmask_all is missing/None")
#             pmask_all = self._last_query_pmask_all  # [B,way,H,W]

#             B, way, H, W = pmask_all.shape
#             idx = gt_way.view(B, 1, 1, 1).expand(-1, 1, H, W)

#             pseudo_prob_gtway = pmask_all.gather(1, idx).squeeze(1)  # [B,H,W] float in [0,1]

#             # -- CHANGED: Use real GT for metrics/visualization during val/test
#             # Use dataset GT for metrics during validation/test if available; otherwise fall back to binarized pseudo.
#             if split in ["val", "test"] and "query_mask" in batch:
#                 gt_mask_bin = batch["query_mask"].long()
#             else:
#                 # training visualization still uses pseudo binarization
#                 if hasattr(self, "binarize_pseudo"):
#                     gt_mask_bin = self.binarize_pseudo(pseudo_prob_gtway)  # [B,H,W] long
#                 else:
#                     gt_mask_bin = (pseudo_prob_gtway > 0.5).long()

#         elif self.sup == "mask":
#             gt_mask_bin = batch["query_mask"].long()
#         else:
#             raise ValueError(f"Unknown sup: {self.sup}")

#         # seg prediction uses predicted way
#         pred_seg = self.predict_mask_from_way(output_masks, pred_way)

#         # ---- VISUALIZATION ----
#         # Works for both "mask" and "pseudo" (requires support_masks exist in batch)
#         if getattr(self.args, "vis", False):
#             self._save_episode_vis_onefile(
#                 batch=batch,
#                 split=split,
#                 batch_idx=batch_idx,
#                 gt_way=gt_way,
#                 pred_seg=pred_seg,
#                 gt_mask_bin=gt_mask_bin,
#                 pseudo_prob=pseudo_prob_gtway,
#                 max_save=50,
#             )

#         # Training loss
#         # In pseudo mode, compute_objective will typically use soft p from _last_query_pmask_all.
#         loss = self.compute_objective(output_cls, output_masks, gt_cls, gt_mask_bin, gt_way)

#         # ---- update meters ----
#         with torch.no_grad():
#             self._cls_correct[split] += float((pred_cls == gt_cls).sum().item())
#             self._cls_total[split] += float(gt_cls.numel())

#             self._loss_sum[split] += float(loss.item())
#             self._loss_n[split] += 1.0

#             if self.avg_meter.get(split, None) is None:
#                 raise RuntimeError(f"avg_meter['{split}'] is None. Did on_{split}_epoch_start run?")

#             # Some AverageMeter implementations read batch["query_mask"].
#             # For pseudo mode, we temporarily provide a binary query_mask for correct IoU.
#             if self.sup == "pseudo":
#                 batch_meter = dict(batch)
#                 batch_meter["query_mask"] = gt_mask_bin
#             else:
#                 batch_meter = batch

#             self.avg_meter[split].update_seg(pred_seg, batch_meter, float(loss.item()))
#             self.log(f"{split}/loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)

#         return loss

#     # -------------------------
#     # epoch ends
#     # -------------------------
#     def training_epoch_end(self, outputs):
#         self._shared_epoch_end("trn")

#     def validation_epoch_end(self, outputs):
#         self._shared_epoch_end("val")

#     def test_epoch_end(self, outputs):
#         self._shared_epoch_end("test")

#     def _shared_epoch_end(self, split):
#         miou = self.avg_meter[split].compute_iou()
#         avg_loss = (self._loss_sum[split] / self._loss_n[split]) if self._loss_n[split] > 0 else 0.0
#         acc = (self._cls_correct[split] / self._cls_total[split]) if self._cls_total[split] > 0 else 0.0
#         acc_pct = acc * 100.0

#         logs = {
#             f"{split}/loss": avg_loss,
#             f"{split}/miou": miou,
#             f"{split}/er": acc_pct,
#         }
#         for k, v in logs.items():
#             self.log(k, v, on_epoch=True, logger=True)





# =========================
# pl_module.py  (FSCSModule)
# =========================
import abc
import os
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image

from common import utils
from common.evaluation import AverageMeter


class FSCSModule(pl.LightningModule, metaclass=abc.ABCMeta):
    """
    LIDC setup:
    - Classification: 5-class on query
      batch["query_class"]: [B] 0..4
      output_cls: [B,5]
    - Segmentation: few-shot binary
      output_masks: [B,way,2,H,W]

    Important for pseudo mode:
      - model must set self._last_query_pmask_all = [B,way,H,W] probabilities.
      - Here we binarize that pseudo mask with an adaptive threshold for metrics/visualization.
      - Training loss is computed inside model.compute_objective (it may use soft p, not this binary mask).
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.way = args.way
        self.sup = args.sup

        self.learner = None

        # meters
        self.avg_meter = {"trn": None, "val": None, "test": None}

        # classification counters
        self._cls_correct = {"trn": 0.0, "val": 0.0, "test": 0.0}
        self._cls_total = {"trn": 0.0, "val": 0.0, "test": 0.0}

        # loss counters
        self._loss_sum = {"trn": 0.0, "val": 0.0, "test": 0.0}
        self._loss_n = {"trn": 0.0, "val": 0.0, "test": 0.0}

    # -------------------------
    # abstract
    # -------------------------
    @abc.abstractmethod
    def forward(self, batch):
        pass

    @abc.abstractmethod
    def train_mode(self):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    @abc.abstractmethod
    def predict_cls_seg(self, batch, nshot):
        pass

    @abc.abstractmethod
    def compute_objective(self, output_cls, output_masks, gt_cls, gt_mask, gt_way, lambda_cls=1.0):
        pass

    # -------------------------
    # helpers
    # -------------------------
    def predict_cls(self, output_cls):
        return torch.argmax(output_cls, dim=1)  # [B]

    def map_class_to_way(self, cls0to4, support_classes_1to5):
        target = cls0to4.long() + 1  # 1..5
        return (support_classes_1to5.long() == target.view(-1, 1)).long().argmax(dim=1)

    def predict_mask_from_way(self, output_masks, way_idx):
        with torch.no_grad():
            B, way, C, H, W = output_masks.shape
            idx = way_idx.view(B, 1, 1, 1, 1).expand(-1, 1, C, H, W)
            seg_logits = output_masks.gather(1, idx).squeeze(1)
            return seg_logits.argmax(dim=1).long()

    # -------------------------
    # VIS helpers
    # -------------------------
    @staticmethod
    def _tensor_img_to_u8(img_chw: torch.Tensor) -> np.ndarray:
        """[3,H,W] -> uint8 [H,W,3] via min-max."""
        x = img_chw.detach().cpu().float()
        x = x.permute(1, 2, 0).numpy()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return (x * 255.0).astype(np.uint8)

    @staticmethod
    def _mask_to_u8(mask_hw: torch.Tensor) -> np.ndarray:
        """[H,W] 0/1 or float -> uint8 [H,W] 0/255 using >0.5."""
        m = mask_hw.detach().cpu().float()
        m = (m > 0.5).to(torch.uint8).numpy()
        return (m * 255).astype(np.uint8)

    @staticmethod
    def _prob_to_u8(prob_hw: torch.Tensor) -> np.ndarray:
        """[H,W] float in [0,1] -> uint8 grayscale [H,W]."""
        p = prob_hw.detach().cpu().float().clamp(0, 1).numpy()
        return (p * 255.0).astype(np.uint8)

    @staticmethod
    def _overlay_red(img_u8: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """Overlay mask on image in RED. img_u8: HWC, mask_u8: HW (0/255)."""
        out = img_u8.astype(np.float32).copy()
        m = (mask_u8 > 0)
        if m.any():
            red = np.zeros_like(out)
            red[..., 0] = 255
            out[m] = (1 - alpha) * out[m] + alpha * red[m]
        return np.clip(out, 0, 255).astype(np.uint8)

    def _save_episode_vis_onefile(
        self,
        batch,
        split: str,
        batch_idx: int,
        gt_way: torch.Tensor,     # [B]
        pred_seg: torch.Tensor,   # [B,H,W]
        gt_mask_bin: torch.Tensor,  # [B,H,W] binary (query GT or binarized pseudo)
        pseudo_prob: torch.Tensor = None,  # [B,H,W] optional
        max_save: int = 50,
    ):
        """
        Saves ONE PNG per batch_idx with panels:
          [support overlay] | [query+GT overlay] | [query+PRED overlay] | (optional) [pseudo prob map]
        """
        if not getattr(self.args, "vis", False):
            return
        if split not in ["val", "test"]:
            return
        if batch_idx >= max_save:
            return
        if hasattr(self, "trainer") and (not self.trainer.is_global_zero):
            return

        out_dir = os.path.join(self.args.logpath, "vis", split)
        os.makedirs(out_dir, exist_ok=True)

        b = 0
        q_img_u8 = self._tensor_img_to_u8(batch["query_img"][b])
        q_gt_u8 = self._mask_to_u8(gt_mask_bin[b])
        q_pr_u8 = self._mask_to_u8(pred_seg[b])

        q_gt_overlay = self._overlay_red(q_img_u8, q_gt_u8)
        q_pr_overlay = self._overlay_red(q_img_u8, q_pr_u8)

        way_idx = int(gt_way[b].item())
        shot_idx = 0
        s_img_u8 = self._tensor_img_to_u8(batch["support_imgs"][b, way_idx, shot_idx])
        s_m_u8 = self._mask_to_u8(batch["support_masks"][b, way_idx, shot_idx])
        s_overlay = self._overlay_red(s_img_u8, s_m_u8)

        panels = [s_overlay, q_gt_overlay, q_pr_overlay]

        if pseudo_prob is not None:
            p_u8 = self._prob_to_u8(pseudo_prob[b])
            p_rgb = np.stack([p_u8, p_u8, p_u8], axis=-1)
            panels.append(p_rgb)

        panel = np.concatenate(panels, axis=1)

        prefix = f"ep{int(self.current_epoch):03d}_b{int(batch_idx):04d}"
        save_path = os.path.join(out_dir, f"{prefix}_SUP_QGT_QPR{'_P' if pseudo_prob is not None else ''}.png")
        Image.fromarray(panel).save(save_path)
        print(f"[VIS] saved: {save_path}")

    # -------------------------
    # epoch starts
    # -------------------------
    def on_train_epoch_start(self):
        utils.fix_randseed(None)
        if getattr(self.trainer, "rerun", False):
            self.trainer.optimizers[0].param_groups[0]["capturable"] = True

        self.avg_meter["trn"] = AverageMeter(self.trainer.train_dataloader.dataset.datasets, self.args.way)

        self._cls_correct["trn"] = 0.0
        self._cls_total["trn"] = 0.0
        self._loss_sum["trn"] = 0.0
        self._loss_n["trn"] = 0.0

        self.train_mode()

    def on_validation_epoch_start(self):
        self._shared_eval_epoch_start(self.trainer.val_dataloaders[0].dataset, split="val")

    def on_test_epoch_start(self):
        test_ds = self.trainer.test_dataloaders[0].dataset
        self._shared_eval_epoch_start(test_ds, split="test")

    def _shared_eval_epoch_start(self, dataset, split: str):
        utils.fix_randseed(0)
        self.avg_meter[split] = AverageMeter(dataset, self.args.way)

        self._cls_correct[split] = 0.0
        self._cls_total[split] = 0.0
        self._loss_sum[split] = 0.0
        self._loss_n[split] = 0.0

        self.eval()

    # -------------------------
    # steps
    # -------------------------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "trn")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "test")

    def _shared_step(self, batch, batch_idx, split):
        output_cls, output_masks = self.forward(batch)  # cls [B,5], masks [B,way,2,H,W]

        if "query_class" not in batch:
            raise KeyError("batch['query_class'] missing (expected 0..4).")
        if "support_classes" not in batch:
            raise KeyError("batch['support_classes'] missing (expected [B,way] 1..5).")

        gt_cls = batch["query_class"].long()
        support_classes = batch["support_classes"].long()

        pred_cls = self.predict_cls(output_cls)
        pred_way = self.map_class_to_way(pred_cls, support_classes)
        gt_way = self.map_class_to_way(gt_cls, support_classes)

        pseudo_prob_gtway = None  # [B,H,W] for visualization if pseudo

        # --------- choose GT seg mask for METRICS/VIS ----------
        # NOTE: training loss can ignore this and use soft pmask inside compute_objective.
        if self.sup == "pseudo":
            if not hasattr(self, "_last_query_pmask_all") or self._last_query_pmask_all is None:
                raise RuntimeError("sup='pseudo' but model._last_query_pmask_all is missing/None")
            pmask_all = self._last_query_pmask_all  # [B,way,H,W]

            B, way, H, W = pmask_all.shape
            idx = gt_way.view(B, 1, 1, 1).expand(-1, 1, H, W)

            pseudo_prob_gtway = pmask_all.gather(1, idx).squeeze(1)  # [B,H,W] float in [0,1]

            # ADAPTIVE threshold binarization (expects model has binarize_pseudo)
            if hasattr(self, "binarize_pseudo"):
                gt_mask_bin = self.binarize_pseudo(pseudo_prob_gtway)  # [B,H,W] long
            else:
                gt_mask_bin = (pseudo_prob_gtway > 0.5).long()

        elif self.sup == "mask":
            gt_mask_bin = batch["query_mask"].long()
        else:
            raise ValueError(f"Unknown sup: {self.sup}")

        # seg prediction uses predicted way
        pred_seg = self.predict_mask_from_way(output_masks, pred_way)

        # ---- VISUALIZATION ----
        # Works for both "mask" and "pseudo" (requires support_masks exist in batch)
        if getattr(self.args, "vis", False):
            self._save_episode_vis_onefile(
                batch=batch,
                split=split,
                batch_idx=batch_idx,
                gt_way=gt_way,
                pred_seg=pred_seg,
                gt_mask_bin=gt_mask_bin,
                pseudo_prob=pseudo_prob_gtway,
                max_save=50,
            )

        # Training loss
        # In pseudo mode, compute_objective will typically use soft p from _last_query_pmask_all.
        loss = self.compute_objective(output_cls, output_masks, gt_cls, gt_mask_bin, gt_way)

        # ---- update meters ----
        with torch.no_grad():
            self._cls_correct[split] += float((pred_cls == gt_cls).sum().item())
            self._cls_total[split] += float(gt_cls.numel())

            self._loss_sum[split] += float(loss.item())
            self._loss_n[split] += 1.0

            if self.avg_meter.get(split, None) is None:
                raise RuntimeError(f"avg_meter['{split}'] is None. Did on_{split}_epoch_start run?")

            # Some AverageMeter implementations read batch["query_mask"].
            # For pseudo mode, we temporarily provide a binary query_mask for correct IoU.
            if self.sup == "pseudo":
                batch_meter = dict(batch)
                batch_meter["query_mask"] = gt_mask_bin
            else:
                batch_meter = batch

            self.avg_meter[split].update_seg(pred_seg, batch_meter, float(loss.item()))
            self.log(f"{split}/loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)

        return loss

    # -------------------------
    # epoch ends
    # -------------------------
    def training_epoch_end(self, outputs):
        self._shared_epoch_end("trn")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end("val")

    def test_epoch_end(self, outputs):
        self._shared_epoch_end("test")

    def _shared_epoch_end(self, split):
        miou = self.avg_meter[split].compute_iou()
        avg_loss = (self._loss_sum[split] / self._loss_n[split]) if self._loss_n[split] > 0 else 0.0
        acc = (self._cls_correct[split] / self._cls_total[split]) if self._cls_total[split] > 0 else 0.0
        acc_pct = acc * 100.0

        logs = {
            f"{split}/loss": avg_loss,
            f"{split}/miou": miou,
            f"{split}/er": acc_pct,
        }
        for k, v in logs.items():
            self.log(k, v, on_epoch=True, logger=True)







