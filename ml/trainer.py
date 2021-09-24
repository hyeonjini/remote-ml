
import utils
import os
import numpy as np
import torch

from importlib import import_module


def classification_training(args):

    # -- settings
    utils.seed_everything(args.seed)
    save_dir = utils.increment_path(os.path.join(args.model_dir, args.name))
    os.mkdir(save_dir)
    #id = save_dir.split('/')[-1]
    id = save_dir.split(os.path.sep)[-1]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- change server state
    utils.request_state_training(id)

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)
    transform = transform_module(
        resize = args.resize,
    )

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_module(
        root=args.data_dir,
        transform=transform,
    )

    # -- data_loader
    train_set, val_set = dataset.train_data, dataset.test_data
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        #num_workers=args.num_workers,
        drop_last=True,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        #num_workers=args.num_workers,
        drop_last=True,
        shuffle=False,
    )
    in_channels = next(iter(val_loader))[0].size(1)
    # -- model
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        in_channels=in_channels,
        num_classes=dataset.num_classes,
    )
    model.to(device)

    # -- loss & metric
    criterion_module = getattr(import_module("loss"), args.criterion)
    criterion = criterion_module()

    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
    )

    # -- logging
    log = utils.get_log_form(id, save_dir, args)

    # -- training

    for epoch in range(args.epochs):

        # train loop
        model.train()
        train_loss_items = []
        train_acc_items = []

        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)
            
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            acc_item = (preds == labels).sum().item()

            train_loss_items.append(loss_item)
            train_acc_items.append(acc_item)

        train_loss = np.sum(train_loss_items) / len(train_loader)
        train_acc = np.sum(train_acc_items) / len(train_set)

        print(
            f"Epoch[{epoch}/{args.epochs}] || "
            f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
        )

        log["train_iter"]["epoch"].append(epoch)
        log["train_iter"]["acc"].append(train_acc)
        log["train_iter"]["loss"].append(train_loss)

        # scheduler.step()
        
        # val loop
        with torch.no_grad():
            model.eval()
            val_loss_items = []
            val_acc_items = []
            
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()

                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)

            torch.save(model, f"{save_dir}/{args.model}_{args.dataset}_{args.task}_{epoch:03}_{val_acc:4.2%}_{val_loss:4.2}.pt")
            print(
                f"Epoch[{epoch}/{args.epochs}] || "
                f"validation loss {val_loss:4.4} || validation accuracy {val_acc:4.2%}"
            )
            log["val_iter"]["epoch"].append(epoch)
            log["val_iter"]["acc"].append(val_acc)
            log["val_iter"]["loss"].append(val_loss)
        
        
        utils.send_to_server(log)

    utils.save_training_result(save_dir, log)

    # -- change server state
    utils.request_state_done(id)

    return log

def segmentation_training():
    pass

def generative_model_training():
    pass