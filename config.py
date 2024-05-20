args = lambda x: None

args.log = True

# data args
args.n_ways = 5
args.n_shots = 1
args.n_queries = 12
args.data_root = 'data'
args.data_aug = True
args.n_test_runs = 600
args.n_aug_support_samples = 5

args.n_cls = 64
args.batch_size = 64
args.test_batch_size = 1
args.num_workers = 2
args.classifier = 'LR'  # [LR, R2-D2]
args.lambd = 0.1  # reg coef in r2-d2

# optimization args
args.learning_rate = 0.01
args.lr_decay_rate = 0.1
args.lr_decay_epochs = [60,80]
args.weight_decay = 0.001
args.momentum = 0.9
args.alpha = 0.01  # coef for ortho-reg and jac reg

# training args
args.epochs = 100
args.save_every = 10
args.eval = 1

# regularizations
args.use_ortho_reg = False
args.use_soc = False
args.use_jac_reg = False

args.jac_reg_type = 'tiny_block'  # ['full', 'block', 'tiny_block']
args.dist = 'rademacher'  # ['rademacher', 'normal']
args.dist_mean = 0
args.dist_std = 1
args.resnet12_block_out_shapes = [[64, 42, 42], [160, 21, 21], [320, 10, 10], [640, 5, 5]]
args.resnet12_tiny_block_out_shapes = [
    [64, 84, 84], [64, 84, 84], [64, 84, 84],
    [160, 42, 42], [160, 42, 42], [160, 42, 42],
    [320, 21, 21], [320, 21, 21], [320, 21, 21],
    [640, 10, 10], [640, 10, 10], [640, 10, 10]
]

# model meta
args.model = 'resnet-12'
args.run_name = args.model + '-base'
args.run_id = '-1'
args.model_name = ''
args.model_path = ''
args.continue_train = False
args.wandb_id = ''
args.last_logged = -1
args.notes = ''
args.wandb_key = ''
