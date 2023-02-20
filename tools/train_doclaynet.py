import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (
    default_argument_parser,
    launch,
)
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from train_net import Trainer, setup


def register_custom_datasets():
    # doclaynet dataset
    DATASET_ROOT = "/kaggle/input/doclaynet/"
    ANN_ROOT = os.path.join(DATASET_ROOT, "COCO")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "PNG")
    VAL_PATH = os.path.join(DATASET_ROOT, "PNG")
    TEST_PATH = os.path.join(DATASET_ROOT, "PNG")
    TRAIN_JSON = os.path.join(ANN_ROOT, "train.json")
    VAL_JSON = os.path.join(ANN_ROOT, "val.json")
    TEST_JSON = os.path.join(ANN_ROOT, "test.json")
    register_coco_instances("doclaynet_train", {}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("doclaynet_val", {}, VAL_JSON, VAL_PATH)
    register_coco_instances("doclaynet_test", {}, TEST_JSON, TEST_PATH)

register_custom_datasets()


def main(args):
    cfg = setup(args)
#     cfg.MODEL.WEIGHTS="/kaggle/input/model-now/model_0171999.pth"
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
