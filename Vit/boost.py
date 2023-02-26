import os
import numpy as np
import torch
import torchvision
import argparse
import random
from modules import transform, network, contrastive_loss, vtcc
from utils import yaml_config_hook, save_model
from torch.utils import data
import modules as criteria

def train():
    loss_epoch = 0
    optimizer.zero_grad()
    for step, ((x_i, x_j, x), _) in enumerate(data_loader):
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        x = x.to('cuda')
        index = torch.arange(step*128,(step+1)*128)
        index = index.to('cuda')
        model.eval()
        with torch.no_grad():
            _, _, _, c = model(x, x)
            pseudo_labels_cur, index_cur = criterion_ins.generate_pseudo_labels(
                c, pseudo_labels[index].to(c.device), index.to(c.device)
            )
            pseudo_labels[index_cur] = pseudo_labels_cur
            pseudo_index = pseudo_labels != -1
        if epoch == args.start_epoch:
            continue

        model.train(True)
        
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_scl = criterion_scl(torch.cat((z_i,z_j),dim=0), c)
        loss_clu = criterion_clu(c_j, pseudo_labels[index]).to('cuda')
        f = torch.cat((z_i,z_j),dim=0)
        h = torch.cat((c_i,c_j),dim=0)
        loss_hcr = criterion_hcr(h,f).to('cuda')

       
        loss = loss_clu + 10*loss_hcr + 0.1*loss_scl
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader)}]\t loss_cluster: {loss_clu.item()} \t loss_hcr: {10*loss_hcr.item()}\t loss_scl: {0.1*loss_scl.item()}")
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
    if args.dataset == "SIRI-WHU":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/SIRI-WHU',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 12
    elif args.dataset == "AID":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/AID',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 30
    elif args.dataset == "DTD":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/DTD',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 47
    elif args.dataset == "UC-Merced":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/UC-Merced/Images',
            transform=transform.Augmentation(size=args.image_size),
        )
        class_num = 21
    else:
        raise NotImplementedError
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
   
    vtcc = vtcc.vit_small()
    model = network.Network_VTCC(vtcc, args.feature_dim, class_num)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    pseudo_labels = -torch.ones(dataset.__len__(), dtype=torch.long)
    loss_device = torch.device("cuda")
    
    if args.reload:
        print("reload training.")
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    criterion_ins = contrastive_loss.InstanceLossBoost(tau=args.instance_temperature, alpha=0.99, gamma=0.5, cluster_num=class_num).to(loss_device)
    criterion_clu = contrastive_loss.ClusterLossBoost(cluster_num=class_num).to(loss_device)
    criterion_hcr = contrastive_loss.HCR().to(loss_device)
    criterion_scl = contrastive_loss.SupConLoss().to(loss_device)
    pseudo_labels = -torch.ones(dataset.__len__(), dtype=torch.long)
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        
        loss_epoch = train()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
    save_model(args, model, optimizer, args.epochs)
