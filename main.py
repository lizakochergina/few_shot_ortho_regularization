from config import args
from dataset import ImageNet
from model import ConvNet
from train import train_epoch, validate_epoch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

save_path = args.run_name + args.run_id

if args.log:
    import wandb
    wandb.login(key='')

    run_name = args.run_name + args.run_id
    run = wandb.init(project='ortho_shot', name=run_name)

    artifact = wandb.Artifact(name=run_name + "_code", type="code")
    artifact.add_file("main.py")
    artifact.add_file("config.py")
    artifact.add_file("train.py")
    artifact.add_file("model.py")
    artifact.add_file("dataset.py")
    artifact.add_file("util.py")
    wandb.log_artifact(artifact)

    print('activated logger')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)

train_set = ImageNet(args, 'train')
val_set = ImageNet(args, 'val')
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)
print('loaded data')

model = ConvNet(num_classes=args.n_cls)
model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    momentum=args.momentum
)

eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)

criterion = nn.CrossEntropyLoss()

print('start training\n')
for epoch in tqdm(range(1, args.epochs+1)):

    scheduler.step()

    train_acc, train_loss = train_epoch(
        train_loader, model, criterion, optimizer, args,
        tqdm_desc=f'training {epoch}/{args.epochs}'
    )
    test_acc, test_acc_top5, test_loss = validate_epoch(
        val_loader, model, criterion,
        tqdm_desc=f'validating {epoch}/{args.epochs}'
    )

    print(f'epoch {epoch}:\ntrain loss {train_loss}\ntrain acc {train_acc}\ntest loss {test_loss}\ntest acc {test_acc}\n')

    if args.log:
        wandb.log({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_acc_top5': test_acc_top5,
            'test_loss': test_loss,
            'lr': scheduler.get_last_lr()[0]
        }, step=epoch)

    if epoch % args.save_every == 0:
        torch.save({
            'epoch': epoch,
            'lr': scheduler.get_last_lr()[0],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path + '_' + str(epoch) + '.pth')
