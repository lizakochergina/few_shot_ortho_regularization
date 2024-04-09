from config import args
from dataset import ImageNet
from model import ConvNet
from resnet import resnet12
from torch.optim import lr_scheduler
from train import train_epoch, validate_epoch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

save_path = args.run_name + args.run_id

if args.log:
    import wandb
    wandb.login(key='')

    run_name = args.run_name + args.run_id
    if args.continue_train:
        run = wandb.init(project='ortho_shot', name=run_name, id=args.wandb_id, resume='must')
    else:
        run = wandb.init(project='ortho_shot', name=run_name, notes=args.notes)

        artifact = wandb.Artifact(name=run_name + "_code", type="code")
        artifact.add_file("main.py")
        artifact.add_file("config.py")
        artifact.add_file("train.py")
        artifact.add_file("resnet.py")
        artifact.add_file("dataset.py")
        artifact.add_file("util.py")
        wandb.log_artifact(artifact)

    print('activated logger')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)

train_set = ImageNet(args, 'train')
val_set = ImageNet(args, 'val')
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
print('loaded data')

last_epoch = 0
if args.continue_train:
    checkpoint = torch.load(args.model_path)
    last_epoch = checkpoint['epoch']
    print('loaded checkpoint, last epoch', last_epoch)

if args.model == 'conv':
    model = ConvNet(num_classes=args.n_cls)
elif args.model == 'resnet-12':
    conv_type = 'soc' if args.use_soc else 'standart'
    print('conv type', conv_type)
    model = resnet12(avg_pool=True, conv_type=conv_type, drop_rate=0.1, dropblock_size=5, num_classes=args.n_cls)

if args.continue_train:
    model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    momentum=args.momentum
)
if args.continue_train:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01806

eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
if args.continue_train:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, 59)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)

criterion = nn.CrossEntropyLoss()

print('start training\n')
for epoch in tqdm(range(last_epoch+1, args.epochs+1)):

    train_acc, train_loss = train_epoch(
        train_loader, model, criterion, optimizer, args,
        tqdm_desc=f'training {epoch}/{args.epochs}'
    )
    test_acc, test_acc_top5, test_loss = None, None, None
    if epoch % args.eval == 0 or epoch == args.epochs:
        test_acc, test_acc_top5, test_loss = validate_epoch(
            val_loader, model, criterion,
            tqdm_desc=f'validating {epoch}/{args.epochs}'
        )

    scheduler.step()
    last_lr = scheduler.get_last_lr()[0]

    print(f'epoch {epoch}:\ntrain loss {train_loss}\ntrain acc {train_acc}\ntest loss {test_loss}\ntest acc {test_acc}\nlr {last_lr}\n\n')

    if args.log and epoch > args.last_logged:
        if epoch % args.eval == 0 or epoch == args.epochs:  
            wandb.log({
                'train_acc': train_acc,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'test_acc_top5': test_acc_top5,
                'test_loss': test_loss,
                'lr': scheduler.get_last_lr()[0]
            }, step=epoch)
        else:
            wandb.log({
                'train_acc': train_acc,
                'train_loss': train_loss,
                'lr': scheduler.get_last_lr()[0]
            }, step=epoch)

    if epoch % args.save_every == 0:
        torch.save({
            'epoch': epoch,
            'lr': scheduler.get_last_lr()[0],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path + '_' + str(epoch) + 'ep.pth')

