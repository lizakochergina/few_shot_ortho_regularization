from util import accuracy, AverageMeter, deconv_orth_dist
from tqdm.notebook import tqdm
import torch


def train_epoch(train_loader, model, criterion, optimizer, args, tqdm_desc=None):
    model.train()

    device = next(model.parameters()).device

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for idx, (input, target, _) in enumerate(tqdm(train_loader, desc=tqdm_desc)):
        input = input.float()
        input = input.to(device)
        target = target.to(device)

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        # ===================stats=====================
        # tracking loss without orthogonal regularization
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================orthogonal regularization====================
        if args.use_ortho_reg:
            diff = deconv_orth_dist(model.layer1[0].weight) + deconv_orth_dist(model.layer2[0].weight) + \
                   deconv_orth_dist(model.layer3[0].weight) + deconv_orth_dist(model.layer4[0].weight)
            loss += diff

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, losses.avg


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
