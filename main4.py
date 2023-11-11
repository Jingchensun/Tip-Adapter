import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from torch.distributions.gamma import Gamma
import json
import wandb
import numpy as np

import renyicl.builder
import renyicl.loader
import renyicl.optimizer
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def run_tip_adapter_F(cfg, test_features, test_labels, clip_weights, clip_model, train_loader_F, template):
    
    model = Adapter(512, 1).to(clip_model.dtype)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    # wandb.config.train_epochs = 100
    best_acc, best_epoch = 0.0, 0
    cfg['train_epoch'] = 100
    for train_idx in range(cfg['train_epoch']): #cfg['train_epoch']
        # Train
        model.train().cuda()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target, text) in enumerate(tqdm(train_loader_F)):
            #print(text[0])
            text = clip.tokenize(text[0]).cuda()
            images, target = images.cuda(), target.cuda()
            # print("target:", target) #torch.Size([256])

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # print("image_features:", image_features.size()) #torch.Size([256, 512])

                text_features = clip_model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            affinity =model(image_features)
            affinity2 = 0.2 * affinity + 0.8 * image_features
            contras_logits =  100. * (affinity2 @ text_features.t())
            # print("clip_logits:", clip_logits)
            groundtruth = torch.arange(len(images), dtype=torch.long).cuda()
            # affinity = adapter(image_features) #cache_keys torch.Size([512, 1616])
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values # cache_values torch.Size([1616, 101])
            cross_logits = 100. * (affinity2 @ clip_weights)

            loss1 = F.cross_entropy(contras_logits, groundtruth)
            loss2 = F.cross_entropy(contras_logits.T, groundtruth)
            loss12 = (loss1 + loss2)/2


            loss3 = F.cross_entropy(cross_logits, target)
            loss = loss3 + loss12


            tip_logits = 100. * (affinity2 @ clip_weights)
            # print("tip_logits:", tip_logits)
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        wandb.log({"epoch": train_idx, "loss12": loss12, "loss3": loss3, "loss": loss,"train_accuracy": acc})
        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        model.eval()
        affinity = model(test_features)
        affinity2 = 0.2 * affinity + 0.8 * test_features
        tip_logits = 100. * (affinity2 @ clip_weights)

        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(model, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    wandb.finish()
    model = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")


    return best_acc


def main():

    # Load config file
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    # CLIP

    clip_model, preprocess = clip.load(cfg['backbone'], device=device)
    clip_model.eval()

    # Prepare dataset
    origin_acc = {}
    for seed in range(3):
        wandb.init(project=("tip-adapter"), entity="jingchensun")
        wandb.run.name = str(cfg['dataset']) + 'seed-'+str(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print("Seed=", seed)

        print("Preparing dataset.")
        dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])

        val_loader = build_data_loader(dataset.template, data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)
        test_loader = build_data_loader(dataset.template, data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess, shuffle=False)

        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        train_loader_cache = build_data_loader(dataset.template, data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
        train_loader_F = build_data_loader(dataset.template, data_source=dataset.train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

        # Textual features
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = clip_classifier(dataset.classnames, dataset.template, clip_model)

        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache)

        # Pre-load val features
        print("\nLoading visual features and labels from val set.")
        val_features, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

        # Pre-load test features
        print("\nLoading visual features and labels from test set.")
        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

        # ------------------------------------------ Tip-Adapter ------------------------------------------
        # run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights)

        # ------------------------------------------ Tip-Adapter-F ------------------------------------------
        best_acc = run_tip_adapter_F(cfg, test_features, test_labels, clip_weights, clip_model, train_loader_F, dataset.template)

        origin_acc[("origin_acc"+str(seed))] = best_acc
    
    file_path = "./output/" + str(cfg['dataset']) + '.json'
    values = list(origin_acc.values())
    mean_accuracy = round(np.mean(values), 3)
    variance_accuracy = round(np.var(values), 3)
    origin_acc["mean"] = mean_accuracy 
    origin_acc["var"] = variance_accuracy
    origin_acc["task"] = "CLIP-Adapter, Crossentropy -D1-ratio-contrastive"
    with open(file_path, 'a',encoding='utf-8') as file:
        json.dump(origin_acc, file, indent=4, ensure_ascii=False)
           

if __name__ == '__main__':
    main()