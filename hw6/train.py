import os
import pandas as pd

from utils.options import args
import utils.common as utils
from model import Model

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR
from transformers import get_cosine_schedule_with_warmup
from dataPreparer import DataPreparation, Data
import wandb

import warnings

warnings.filterwarnings("ignore")

# device = torch.device(f"cuda:{args.gpus[0]}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = utils.checkpoint(args)


def main():
    start_epoch = 0
    best_acc = 0.0

    # Data loading
    print('=> Preparing data..')

    # data loader

    loader = Data(args, data_path=args.src_data_path, label_path=args.src_label_path)

    data_loader = loader.loader_train
    data_loader_valid = loader.loader_valid
    data_loader_test = loader.loader_test

    # Create model
    print('=> Building model...')

    # load training model
    # model = import_module(f'model.{args.arch}').__dict__[
    #     args.model]().to(device)
    model = Model().to(device)

    # Load pretrained weights
    if args.pretrained:
        ckpt = torch.load(os.path.join(checkpoint.ckpt_dir,
                          args.source_file), map_location=device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)

    if args.inference_only:
        inference(args, data_loader_test, model, args.output_file)
        return

    param = [param for name, param in model.named_parameters()]

    optimizer = optim.AdamW(param, lr=args.lr, weight_decay=args.weight_decay)
    total_step = len(data_loader) * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(
        0.05 * total_step), num_training_steps=total_step-int(0.05 * total_step))

    for epoch in range(start_epoch, args.num_epochs):
        train(args, data_loader, model, optimizer, epoch, scheduler)
        valid_acc = valid(args, data_loader_valid, model)

        is_best = best_acc < valid_acc
        best_acc = max(best_acc, valid_acc)

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    inference(args, data_loader_test, model, args.output_file)
    print(f'Best acc: {best_acc:.3f}\n')


def train(args, data_loader, model, optimizer, epoch, scheduler):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()
    criterion = nn.CrossEntropyLoss()
    num_iterations = len(data_loader)

    # switch to train mode
    model.train()

    for i, (inputs, targets, _) in enumerate(data_loader, 1):

        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # train
        output = model(inputs)
        loss = criterion(output, targets)

        # optimize cnn
        loss.backward()
        optimizer.step()
        scheduler.step()

        # train weights
        losses.update(loss.item(), inputs.size(0))

        # evaluate
        prec1, _ = utils.accuracy(output, targets, topk=(1, 5))
        acc.update(prec1[0], inputs.size(0))

        if args.wandb:
            wandb.log(
                {
                    "Train Loss val": losses.val,
                    "Train Loss avg": losses.avg,
                    "Train Acc val": acc.val,
                    "Train Acc avg": acc.avg,
                }
            )

        if i % args.print_freq == 0:
            print(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(
                    epoch, i, num_iterations,
                    train_loss=losses,
                    acc=acc))


def valid(args, loader_valid, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_valid, 1):

            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = criterion(preds, targets)

            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec1[0], inputs.size(0))

    print(f'Validation acc {acc.avg:.3f}\n')

    return acc.avg


def inference(args, loader_test, model, output_file_name):
    outputs = []
    datafiles = []
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            _, output = preds.topk(1, 1, True, True)

            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))

            datafiles.extend(list(datafile))

    # print(output)
    output_file = dict()
    output_file['image_name'] = datafiles
    output_file['label'] = outputs

    output_file = pd.DataFrame.from_dict(output_file)
    output_file.to_csv(output_file_name, index=False)


if __name__ == '__main__':
    wandb_config = vars(args)
    run = wandb.init(
        project=f"DS_HW6",
        config=wandb_config,
        group="MNISTM",
    )
    artifact = wandb.Artifact("model", type="model")
    main()