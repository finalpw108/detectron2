from tools import train_net

def train_multi_gpu():
     args = default_argument_parser().parse_args()
     launch(
        main,
        num_gpus_per_machine=2,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(args,),
    )
