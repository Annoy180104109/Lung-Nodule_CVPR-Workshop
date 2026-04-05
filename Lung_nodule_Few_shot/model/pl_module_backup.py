import abc
import torch
import pytorch_lightning as pl

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
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.way = args.way
        self.sup = args.sup

        self.learner = None
        self.avg_meter = {"trn": None, "val": None}

        # classification counters
        self._cls_correct = {"trn": 0.0, "val": 0.0}
        self._cls_total = {"trn": 0.0, "val": 0.0}

        # loss counters
        self._loss_sum = {"trn": 0.0, "val": 0.0}
        self._loss_n = {"trn": 0.0, "val": 0.0}

    # abstract
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

    # helpers
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

    # epoch starts
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
        self._shared_eval_epoch_start(self.trainer.val_dataloaders[0].dataset)

    def _shared_eval_epoch_start(self, dataset):
        utils.fix_randseed(0)
        self.avg_meter["val"] = AverageMeter(dataset, self.args.way)

        self._cls_correct["val"] = 0.0
        self._cls_total["val"] = 0.0
        self._loss_sum["val"] = 0.0
        self._loss_n["val"] = 0.0

        self.eval()

    # steps
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "trn")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, batch_idx, "val")

    def _shared_step(self, batch, batch_idx, split):
        output_cls, output_masks = self.forward(batch)  # cls [B,5], masks [B,way,2,H,W]

        if "query_class" not in batch:
            raise KeyError("batch['query_class'] missing (expected 0..4).")
        if "support_classes" not in batch:
            raise KeyError("batch['support_classes'] missing (expected [B,way] 1..5).")

        gt_cls = batch["query_class"].long()              # [B]
        support_classes = batch["support_classes"].long() # [B,way]

        pred_cls = self.predict_cls(output_cls)           # [B]
        pred_way = self.map_class_to_way(pred_cls, support_classes)
        gt_way = self.map_class_to_way(gt_cls, support_classes)

        # --------- choose GT seg mask ---------
        if self.sup == "pseudo":
            # model must provide _last_query_pmask_all: [B,way,H,W]
            if not hasattr(self, "_last_query_pmask_all") or self._last_query_pmask_all is None:
                raise RuntimeError("sup='pseudo' but model._last_query_pmask_all is missing/None")
            pmask_all = self._last_query_pmask_all  # [B,way,H,W]

            B, way, H, W = pmask_all.shape
            idx = gt_way.view(B, 1, 1, 1).expand(-1, 1, H, W)
            gt_mask = pmask_all.gather(1, idx).squeeze(1)  # [B,H,W]
        elif self.sup == "mask":
            gt_mask = batch["query_mask"]
        else:
            raise ValueError(f"Unknown sup: {self.sup}")

        # seg prediction uses predicted way
        pred_seg = self.predict_mask_from_way(output_masks, pred_way)

        loss = self.compute_objective(output_cls, output_masks, gt_cls, gt_mask, gt_way)

        with torch.no_grad():
            self._cls_correct[split] += float((pred_cls == gt_cls).sum().item())
            self._cls_total[split] += float(gt_cls.numel())

            self._loss_sum[split] += float(loss.item())
            self._loss_n[split] += 1.0

            self.avg_meter[split].update_seg(pred_seg, batch, float(loss.item()))

            self.log(f"{split}/loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)

        return loss

    # epoch ends
    def training_epoch_end(self, outputs):
        self._shared_epoch_end("trn")

    def validation_epoch_end(self, outputs):
        self._shared_epoch_end("val")

    def _shared_epoch_end(self, split):
        miou = self.avg_meter[split].compute_iou()
        avg_loss = (self._loss_sum[split] / self._loss_n[split]) if self._loss_n[split] > 0 else 0.0
        acc = (self._cls_correct[split] / self._cls_total[split]) if self._cls_total[split] > 0 else 0.0
        acc_pct = acc * 100.0

        logs = {
            f"{split}/loss": avg_loss,
            f"{split}/miou": miou,
            f"{split}/er": acc_pct,  # accuracy (%)
        }
        for k, v in logs.items():
            self.log(k, v, on_epoch=True, logger=True)