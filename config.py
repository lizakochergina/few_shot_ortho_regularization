args = lambda x: None

args.log = False

# data args
args.n_ways = 5
args.n_shots = 1
args.n_queries = 12
args.data_root = 'data'
args.data_aug = False
args.n_test_runs = 5
args.n_aug_support_samples = 1

args.n_cls = 64
args.batch_size = 64
args.test_batch_size = 1
args.num_workers = 2

# optimization args
args.learning_rate = 1e-3  # 1e-4 in paper
args.lr_decay_rate = 0.1
args.lr_decay_epochs = [60,80]  # check
args.weight_decay = 2e-5
args.momentum = 0.9

# training args
args.epochs = 100
args.save_every = 10
args.run_name = 'base-classifier'
args.run_id = '-2'
args.use_ortho_reg = False
args.model_path = ''