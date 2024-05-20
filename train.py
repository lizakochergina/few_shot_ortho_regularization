from util import accuracy, AverageMeter, deconv_orth_dist, orth_dist
from ortho_vec_util import generate_random_vectors, orthogonal_loss
from tqdm import tqdm
import torch


def train_epoch(train_loader, model, criterion, optimizer, args,
                ort_vectors=None, eff_ranks=None, tqdm_desc=None):
    model.train()

    device = next(model.parameters()).device

    orth_loss = AverageMeter()
    cum_loss = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (input, target, _) in enumerate(tqdm(train_loader, desc=tqdm_desc)):
        input = input.float()
        input = input.to(device)
        target = target.to(device)

        # ===================forward=====================
        [f0, f1, f2, f3, feat], output = model(input, is_feat=True)
        loss = criterion(output, target)

        # ===================stats=====================
        # tracking loss without orthogonal regularization
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================orthogonal regularization====================
        if args.use_ortho_reg and args.model == 'conv':
            diff = deconv_orth_dist(model.layer1[0].weight) + deconv_orth_dist(model.layer2[0].weight) + \
                   deconv_orth_dist(model.layer3[0].weight) + deconv_orth_dist(model.layer4[0].weight)
            loss += diff
        elif args.use_ortho_reg and args.model == 'resnet-12':
            diff = orth_dist(model.layer2[0].downsample[0].weight) + \
                   orth_dist(model.layer3[0].downsample[0].weight) + \
                   orth_dist(model.layer4[0].downsample[0].weight)

            diff += deconv_orth_dist(model.layer1[0].conv1.weight, stride=1) + \
                    deconv_orth_dist(model.layer1[0].conv3.weight, stride=1)

            diff += deconv_orth_dist(model.layer2[0].conv1.weight, stride=1) + \
                    deconv_orth_dist(model.layer2[0].conv3.weight, stride=1)

            diff += deconv_orth_dist(model.layer3[0].conv1.weight, stride=1) + \
                    deconv_orth_dist(model.layer3[0].conv3.weight, stride=1)

            diff += deconv_orth_dist(model.layer4[0].conv1.weight, stride=1) + \
                    deconv_orth_dist(model.layer4[0].conv3.weight, stride=1)

            loss += args.alpha * diff
        elif args.use_jac_reg:
            internal_input = [input, f0, f1, f2]
            loss_ = orthogonal_loss(model, ort_vectors, internal_input, args.jac_reg_type)
            orth_loss.update(loss_.item())
            loss += args.alpha * loss_.item()

        # ===================backward=====================
        cum_loss.update(loss.item(), input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, losses.avg, orth_loss.avg, cum_loss.avg


@torch.no_grad()
def validate_epoch(val_loader, model, criterion, tqdm_desc=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    device = next(model.parameters()).device

    # switch to evaluate mode
    model.eval()

    for idx, (input, target, _) in enumerate(tqdm(val_loader, desc=tqdm_desc)):
        input = input.float()
        input = input.to(device)
        target = target.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

    return top1.avg, top5.avg, losses.avg
