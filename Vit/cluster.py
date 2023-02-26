import os
import argparse
import torch
import torchvision
import numpy as np
from utils import yaml_config_hook
from modules import network, transform, vtcc
from evaluation import evaluation
from torch.utils import data
import copy


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.backends.cudnn.deterministic=True
    if args.dataset == "UC-Merced":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/UC-Merced/Images',
            transform=transform.Augmentation(size=args.image_size).test_transform,
        )
        class_num = 21
    elif args.dataset == "SIRI-WHU":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/SIRI-WHU',
            transform=transform.Augmentation(size=args.image_size).test_transform,
        )
        class_num = 12
    elif args.dataset == "AID":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/AID',
            transform=transform.Augmentation(size=args.image_size).test_transform,
        )
        class_num = 30
    elif args.dataset == "DTD":
        dataset = torchvision.datasets.ImageFolder(
            root='./datasets/DTD',
            transform=transform.Augmentation(size=args.image_size).test_transform,
        )
        class_num = 47
    else:
        raise NotImplementedError
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )


    vtcc = vtcc.vit_small()
    model = network.Network_VTCC(vtcc, args.feature_dim, class_num)
    model = model.to('cuda')

    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device)
    nmi, ari, f, acc = evaluation.evaluate(Y, X)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'.format(nmi, ari, f, acc))
