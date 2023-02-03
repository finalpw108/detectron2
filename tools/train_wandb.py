import os
import wandb
from kaggle_secrets import UserSecretsClient
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

register_custom_datasets()


def main(args,config):
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
  
  def train_wandb():
        sweep_config = {
        'method': 'bayes'  # random, grid
        }


        metric = {  
            'name': 'total_loss',  
            'goal': 'minimize'
        }
        sweep_config['metric'] = metric


        parameters_dict = {

            'learning_rate': {  
                'distribution': 'uniform',  
                'min' : 0,
                'max' : 0.001
            },
            'IMS_PER_BATCH': {
                'values': [4, 8, 12]
            },
            'iteration': {
                'values': [300, 1000, 1500, 5000]
            },
            'BATCH_SIZE_PER_IMAGE' : {
                'values': [32, 64, 128, 256]
            },    
            'model': {
                'value' : 'Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml'
            }


        }
        sweep_config['parameters'] = parameters_dict
        user_secrets = UserSecretsClient()
        api = user_secrets.get_secret("wandb-key")
        wandb.login(key=api)
        sweep_id = wandb.sweep(sweep_config, project='document_analysis')
         with wandb.init(project='document_analysis', config=config) as run:
            config = wandb.config  
            cfg = main(config)
    


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
