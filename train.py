from modules.dataset import DataClass
from modules.utils import weights_init, warm_up_and_annealing
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from modules.train_scripts import train
import argparse
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import pickle
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./data')
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/')
    parser.add_argument('--writer_path', type=str, default='./logs/', help='path to save tensorboard logs')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='total number of epochs')
    parser.add_argument('--warm_up_epochs', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start from when training is resumed')
    parser.add_argument('--supervised_ratio', type=float, default=1, help='ratio of supervised data')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_name', type=str, default='efficientnet-b0')
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--experiment_name', type=str, default='default_experiment')
    parser.add_argument('--resume_training', type=bool, default=False)
    args = parser.parse_args()

    #######################################################################
    if not os.path.isdir(args.checkpoints_path + args.experiment_name):
        os.mkdir(args.checkpoints_path + args.experiment_name)

    DEVICE = torch.device(args.device)

    train_dataset = DataClass(args.dataset_path, mode='train')
    val_dataset = DataClass(args.dataset_path, mode='val')
    with open(args.checkpoints_path + args.experiment_name + '/label_encoder.pkl', 'wb') as le_dump_file:
        pickle.dump(train_dataset.label_encoder, le_dump_file)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.pretrained:
        model = EfficientNet.from_pretrained(args.model_name)
    else:
        model = EfficientNet.from_name(args.model_name)

    model._fc = nn.Linear(in_features=1280, out_features=args.num_classes, bias=True)
    model = model.to(DEVICE)
    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: warm_up_and_annealing(
                                                                    it, args.warm_up_epochs, args.epochs, train_loader, args.resume_training))
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=args.writer_path + args.experiment_name)

    if args.resume_training:
        print('ha')
        checkpoint = torch.load(args.checkpoints_path + args.experiment_name + '/checkpoints.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        model.train()

    train(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=args.epochs,
          start_epoch=args.start_epoch, writer=writer, device=DEVICE,
          checkpoints_path=args.checkpoints_path + args.experiment_name)

