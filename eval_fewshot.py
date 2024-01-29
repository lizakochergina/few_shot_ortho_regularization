import time

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model import ConvNet
# from models import model_dict, model_pool
# from models.util import create_model

from dataset import ImageNet, MetaImageNet
# from dataset.transform_cfg import transforms_options, transforms_list

from meta_eval import meta_test

from config import args
from eval_util import R2D2


def main():

    # test loader
    args.batch_size = args.test_batch_size
    # args.n_aug_support_samples = 1

    meta_testloader = DataLoader(MetaImageNet(args=args, partition='test', fix_seed=False),
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
    meta_valloader = DataLoader(MetaImageNet(args=args, partition='val', fix_seed=False),
                                batch_size=args.test_batch_size, shuffle=False, drop_last=False,
                                num_workers=args.num_workers)

    # load model
    model = ConvNet(num_classes=args.n_cls)
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model_state_dict'])

    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # evalation
    if args.classifier == 'R2-D2':
        base_learner = R2D2(init_scale=1e-4, init_bias=0, learn_lambda=False, init_lambda=1)
        opt = torch.optim.SGD([base_learner.adjust_layer, base_learner.lambda_layer], lr=0.01)
    else:
        base_learner = None
        opt = None

    start = time.time()
    val_acc, val_std = meta_test(model, meta_valloader, opt=opt, r2d2_learner=base_learner, mode='train')
    val_time = time.time() - start
    print('val_acc: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc, val_std,
                                                                  val_time))

    start = time.time()
    test_acc, test_std = meta_test(model, meta_testloader, opt=opt, r2d2_learner=base_learner, mode='test')
    test_time = time.time() - start
    print('test_acc: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc, test_std,
                                                                    test_time))

    if args.classifier == 'R2-D2':
        base_learner = R2D2(init_scale=1e-4, init_bias=0, learn_lambda=False, init_lambda=1)
        opt = torch.optim.SGD([base_learner.adjust_layer, base_learner.lambda_layer], lr=0.01)
    else:
        base_learner = None
        opt = None

    start = time.time()
    val_acc_feat, val_std_feat = meta_test(model, meta_valloader, use_logit=False, opt=opt, r2d2_learner=base_learner, mode='train')
    val_time = time.time() - start
    print('val_acc_feat: {:.4f}, val_std: {:.4f}, time: {:.1f}'.format(val_acc_feat,
                                                                       val_std_feat,
                                                                       val_time))

    start = time.time()
    test_acc_feat, test_std_feat = meta_test(model, meta_testloader, use_logit=False, opt=opt, r2d2_learner=base_learner, mode='test')
    test_time = time.time() - start
    print('test_acc_feat: {:.4f}, test_std: {:.4f}, time: {:.1f}'.format(test_acc_feat,
                                                                         test_std_feat,
                                                                         test_time))


if __name__ == '__main__':
    main()