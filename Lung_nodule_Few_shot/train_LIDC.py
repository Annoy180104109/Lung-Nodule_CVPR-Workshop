# import os
# import argparse
# import torch

# from pytorch_lightning import Trainer

# from model.cst import ClfSegTransformer
# from data.datasetlload import FSCSDatasetModule
# from common.callbacks import CustomCheckpoint, OnlineLogger, CustomProgressBar


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser(
#         description='Classification Segmentation Transformer for Few-Shot Classification and Segmentation'
#     )

#     parser.add_argument(
#         '--datapath',
#         type=str,
#         default='/home/g202417400/Msproject/MSPO-Net/cst/root/LIDC_full_annotation',
#     )

#     parser.add_argument(
#         '--benchmark',
#         type=str,
#         default='lidc',
#         choices=['pascal', 'coco', 'lidc'],
#     )

#     # ✅ updated: add resnet101
#     parser.add_argument(
#         '--backbone',
#         type=str,
#         default='resnet101',
#         choices=['vit-small', 'resnet50', 'resnet101'],
#     )

#     parser.add_argument('--logpath', type=str, default='/home/g202417400/Msproject/MSPO-Net/cst/loglidc')

#     parser.add_argument('--way', type=int, default=1)
#     parser.add_argument('--shot', type=int, default=1)

#     parser.add_argument('--batchsize', type=int, default=12)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--maxepochs', type=int, default=50)

#     parser.add_argument('--fold', type=int, default=0)

#     parser.add_argument('--nowandb', action='store_true')
#     parser.add_argument('--eval', action='store_true')
#     parser.add_argument('--sup', type=str, default='mask', choices=['mask', 'pseudo'])
#     parser.add_argument('--resume', action='store_true')
#     parser.add_argument('--vis', action='store_true')
#     parser.add_argument('--imgsize', type=int, default=96)

#     # ⭐ NEW VARIABLE (ONLY THIS CONTROLS MIXED SUPERVISION)
#     parser.add_argument(
#         '--pseudo_ratio',
#         type=float,
#         default=0.6,
#         help='Fraction of query pixels supervised by pseudo mask (0.0–1.0)'
#     )

#     args = parser.parse_args()

#     dm = FSCSDatasetModule(args, img_size=args.imgsize)
#     ckpt_callback = CustomCheckpoint(args)

#     trainer = Trainer(
#         strategy="dp",
#         gpus=torch.cuda.device_count(),
#         callbacks=[ckpt_callback, CustomProgressBar(args)],
#         logger=False if args.nowandb or args.eval else OnlineLogger(args),
#         max_epochs=args.maxepochs,
#         num_sanity_val_steps=0,
#         default_root_dir=ckpt_callback.modelpath,
#         enable_checkpointing=True,
#     )

#     if args.eval:
#         modelpath = ckpt_callback.modelpath
#         model = ClfSegTransformer.load_from_checkpoint(modelpath, args=args)
#         trainer.test(model, dataloaders=dm.test_dataloader())
#     else:
#         model = ClfSegTransformer(args)

#         if os.path.exists(ckpt_callback.lastmodelpath):
#             ckpt_path = ckpt_callback.lastmodelpath
#             trainer.rerun = True
#         else:
#             ckpt_path = None
#             trainer.rerun = False

#         trainer.fit(model, dm, ckpt_path=ckpt_path)



import os
import argparse
import torch

from pytorch_lightning import Trainer

from model.cst import ClfSegTransformer
from data.datasetlload import FSCSDatasetModule
from common.callbacks import CustomCheckpoint, OnlineLogger, CustomProgressBar


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Classification Segmentation Transformer for Few-Shot Classification and Segmentation'
    )

    parser.add_argument(
        '--datapath',
        type=str,
        default='/home/g202417400/Msproject/MSPO-Net/cst/root/LIDC_full_annotation',
    )

    parser.add_argument(
        '--benchmark',
        type=str,
        default='lidc',
        choices=['pascal', 'coco', 'lidc'],
    )

    parser.add_argument('--backbone', type=str, default='vit-small')
    parser.add_argument('--logpath', type=str, default='/home/g202417400/Msproject/MSPO-Net/cst/loglidc')

    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)

    parser.add_argument('--batchsize', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--maxepochs', type=int, default=5)

    parser.add_argument('--fold', type=int, default=0)

    parser.add_argument('--nowandb', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sup', type=str, default='mask', choices=['mask', 'pseudo'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--imgsize', type=int, default=96)

    # ⭐ NEW VARIABLE (ONLY THIS CONTROLS MIXED SUPERVISION)
    parser.add_argument(
        '--pseudo_ratio',
        type=float,
        default=1.0,
        help='Fraction of query pixels supervised by pseudo mask (0.0–1.0)'
    )

    args = parser.parse_args()

    dm = FSCSDatasetModule(args, img_size=args.imgsize)
    ckpt_callback = CustomCheckpoint(args)

    trainer = Trainer(
        strategy="dp",
        gpus=torch.cuda.device_count(),
        callbacks=[ckpt_callback, CustomProgressBar(args)],
        logger=False if args.nowandb or args.eval else OnlineLogger(args),
        max_epochs=args.maxepochs,
        num_sanity_val_steps=0,
        default_root_dir=ckpt_callback.modelpath,
        enable_checkpointing=True,
    )

    if args.eval:
        modelpath = ckpt_callback.modelpath
        model = ClfSegTransformer.load_from_checkpoint(modelpath, args=args)
        trainer.test(model, dataloaders=dm.test_dataloader())
    else:
        model = ClfSegTransformer(args)

        if os.path.exists(ckpt_callback.lastmodelpath):
            ckpt_path = ckpt_callback.lastmodelpath
            trainer.rerun = True
        else:
            ckpt_path = None
            trainer.rerun = False

        trainer.fit(model, dm, ckpt_path=ckpt_path)