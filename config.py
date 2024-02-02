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
args.classifier = 'R2-D2'  # [LR, R2-D2]
args.lambd = 0.1  # reg coef in r2-d2

args.n_cls = 64
args.batch_size = 64
args.test_batch_size = 1
args.num_workers = 2

# optimization args
args.learning_rate = 0.05  # 1e-4 in paper
args.lr_decay_rate = 0.1
args.lr_decay_epochs = [60,80]  # check
args.weight_decay = 5e-4
args.momentum = 0.9
args.alpha = 0.01  # coef in ortho reg

# training args
args.epochs = 100
args.save_every = 10
args.model = 'resnet-12'
args.run_name = args.model + '-ortho-reg'
args.run_id = '-15'
args.use_ortho_reg = True
args.model_path = ''
args.continue_train = False
args.wandb_id = ''
args.last_logged = 0
args.notes = ''
