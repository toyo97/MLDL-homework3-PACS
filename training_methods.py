import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from net import dann


def loopy(dl):
    while True:
        for x in iter(dl): yield x


def update_ratio(outputs, labels, current_ratio):
    _, preds = torch.max(outputs.data, 1)
    current_ratio[0] += torch.sum(labels.data == preds).data.item()
    current_ratio[1] += preds.size()[0]


def source_only_train(source_dir='HW3/PACS/photo', target_dir='HW3/PACS/art_painting', batch_size=64, num_classes=7,
                      lr=1e-2, momentum=0.9, weight_decay=5e-5, epochs=20, step_size=10, gamma=0.1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # Define transforms for training phase
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(imgnet_mean, imgnet_std)
                                          ])
    # Define transforms for the evaluation phase
    eval_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(imgnet_mean, imgnet_std)
                                         ])
    source_dataset = torchvision.datasets.ImageFolder(source_dir, transform=train_transform)
    target_dataset = torchvision.datasets.ImageFolder(target_dir, transform=eval_transform)
    source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    net = dann(pretrained=True, progress=True, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # for both classifier and discriminator
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    num_batches = len(source_dataset) // batch_size
    net = net.to(device)
    cudnn.benchmark = True  # optimizes runtime
    src_accs = []
    tgt_accs = []
    losses = []
    for _ in tqdm(range(epochs)):

        net.train()  # training mode
        source_iter = iter(source_dataloader)
        acc_ratio = [0, 0]  # initialize accuracy ratio
        for it in range(num_batches):

            optimizer.zero_grad()  # zero-ing the gradients

            images, labels = next(source_iter)
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            update_ratio(outputs, labels, acc_ratio)

            loss = criterion(outputs, labels)

            if it == num_batches - 1:
                losses.append(loss.item())

            loss.backward()
            optimizer.step()  # update weights based on accumulated gradients

        src_accs.append(acc_ratio[0] / acc_ratio[1])

        scheduler.step()

        net.train(False)  # Set Network to evaluation mode

        acc_ratio = [0, 0]
        for images, labels in target_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            update_ratio(outputs, labels, acc_ratio)

        tgt_accs.append(acc_ratio[0] / acc_ratio[1])

    return tgt_accs


def DA_train(source_dir='HW3/PACS/photo', target_dir='HW3/PACS/art_painting', batch_size=64, num_classes=7,
             lr=1e-2, momentum=0.9, weight_decay=5e-5, epochs=20, step_size=10, gamma=0.1, alpha=0.1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # Define transforms for training phase
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(imgnet_mean, imgnet_std)
                                          ])
    # Define transforms for the evaluation phase
    eval_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(imgnet_mean, imgnet_std)
                                         ])
    source_dataset = torchvision.datasets.ImageFolder(source_dir, transform=train_transform)
    target_dataset = torchvision.datasets.ImageFolder(target_dir, transform=eval_transform)
    source_dataloader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    net = dann(pretrained=True, progress=True, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # for both classifier and discriminator
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    num_batches = max(len(source_dataset), len(target_dataset)) // batch_size
    net = net.to(device)
    cudnn.benchmark = True  # optimizes runtime
    src_accs = []
    tgt_accs = []
    losses_y = []
    losses_d = []
    for _ in tqdm(range(epochs)):

        source_iter = loopy(source_dataloader)
        target_iter = loopy(target_dataloader)
        acc_ratio = [0, 0]  # initialize accuracy ratio
        for it in range(num_batches):
            net.train()  # training mode
            optimizer.zero_grad()  # zero-ing the gradients

            # ************ #
            # SOURCE to Gy #
            # ************ #
            images, labels = next(source_iter)
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            update_ratio(outputs, labels, acc_ratio)

            loss_y = criterion(outputs, labels)
            loss_y.backward()

            # ************ #
            # SOURCE to Gd #
            # ************ #
            # same images of previous forward pass, different labels (domain label)
            labels = torch.zeros(images.size()[0], device=device, dtype=torch.long)

            outputs = net(images, alpha=alpha)

            # computing loss for source in discriminator
            loss_sd = criterion(outputs, labels)
            loss_sd.backward()

            # ************ #
            # TARGET to Gd #
            # ************ #
            images, _ = next(target_iter)
            images = images.to(device)
            labels = torch.ones(images.size()[0], device=device, dtype=torch.long)

            outputs = net(images, alpha=alpha)

            # computing loss for source in discriminator
            loss_td = criterion(outputs, labels)
            loss_td.backward()

            if it == num_batches - 1:
                losses_y.append(loss_y.item())
                losses_d.append((loss_td + loss_sd).item())

            optimizer.step()  # update weights based on accumulated gradients

        src_accs.append(acc_ratio[0] / acc_ratio[1])

        scheduler.step()

        net.train(False)  # Set Network to evaluation mode

        acc_ratio = [0, 0]
        for images, labels in target_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            update_ratio(outputs, labels, acc_ratio)

        tgt_accs.append(acc_ratio[0] / acc_ratio[1])

    return tgt_accs

