import os
import numpy as np
import torch
import torchvision
import argparse
import random
from modules import transform, resnet, network, contrastive_loss, transform_s_w
from utils import yaml_config_hook, save_model
from torch.utils import data


def train():
    loss_epoch = 0
    for step, ((x_i, x_j, x), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        x = x.to('cuda')
        
        model.eval()
        with torch.no_grad():
            c = model.forward_logits(x)
        model.train(True)
        
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_pcl = criterion_scl(torch.cat((z_i,z_j),dim=0), c)
        f = torch.cat((z_i,z_j),dim=0)
        h = torch.cat((c_i,c_j),dim=0)
        loss_bce = criterion_hcr(h,f).to('cuda')

       
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster + 10*loss_bce + 0.1*loss_pcl
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()} \t loss_cluster: {loss_cluster.item()} \t loss_hcr: {10*loss_hcr.item()}\t loss_scl: {0.1*loss_instance.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.backends.cudnn.deterministic=True; np.random.seed(args.seed); random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # prepare data
    if args.dataset == "CIFAR-10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=True,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            download=True,
            train=False,
            transform=transform.Transforms(size=args.image_size, s=0.5),
        )
        # train_dataset = torchvision.datasets.CIFAR10(
        #     root=args.dataset_dir,
        #     download=False,
        #     train=True,
        #     transform=transform_s_w.Augmentation(),
        # )
        # test_dataset = torchvision.datasets.CIFAR10(
        #     root=args.dataset_dir,
        #     download=False,
        #     train=False,
        #     transform=transform_s_w.Augmentation(),
        # )
        dataset = data.ConcatDataset([train_dataset, test_dataset])
        class_num = 10
    elif args.dataset == "ImageNet-10":
        dataset = torchvision.datasets.ImageFolder(
            root='datasets/imagenet-10',
            transform=transform.Transforms(size=args.image_size, blur=True),
        )
        class_num = 10
    else:
        raise NotImplementedError
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    criterion_bce = contrastive_loss.BCELoss().to(loss_device)
    criterion_pcl = contrastive_loss.PCLLoss().to(loss_device)

    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
