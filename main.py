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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
a_u = 1
b_u = 1
a_minus = 10
b_minus = 1
a_plus = 5
b_plus = 1


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights):
    
    print("\n-------- Searching hyperparameters on the val set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = val_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights)


    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    clip_logits = 100. * test_features @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

def sample_w(U, s_matrix):
    BS = s_matrix.shape[0]
    s_plus = s_matrix.masked_select(torch.eye(BS).bool().to(device))
    s_minus = s_matrix.masked_select(~torch.eye(BS).bool().to(device))

    w_plus_dist = Gamma(torch.tensor(1+a_plus).float().to(device), U*s_plus + b_plus)
    U = U.repeat_interleave(int(BS-1))
    w_minus_dist = Gamma(torch.tensor(a_minus).float().to(device), U*s_minus + b_minus)

    w_plus = w_plus_dist.sample()
    w_minus = w_minus_dist.sample()

    result = torch.zeros(BS, BS).to(device)
    diagonal_matrix = torch.diag(w_plus)

    result += diagonal_matrix
    mask = ~torch.eye(BS, dtype=bool)  
    result[mask] = w_minus
    # print("w_matrix:", result)
    return result

def sample_u(w_matrix, sim_matrix):
    full_mat = w_matrix * sim_matrix
    rate_param = b_u + full_mat.sum(dim=1)
    u_dist = Gamma(torch.tensor(a_u).float().to(device),\
            rate_param.float())
    # print(rate_param)
    # print(u_dist.sample())
    return u_dist.sample()

def run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, template):
    

    # Enable the cached keys to be learnable
    adapter = nn.Linear(512, 512, bias=True).to(clip_model.dtype).cuda()
    # adapter.weight = nn.Parameter(cache_keys.t())
    torch.nn.init.kaiming_normal_(adapter.weight)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']): #cfg['train_epoch']
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target, text) in enumerate(tqdm(train_loader_F)):
            # print(text[0])
            text = clip.tokenize(text[0]).cuda()
            images, target = images.cuda(), target.cuda()
            # print("target:", target) #torch.Size([256])

            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                print("image_features:", image_features.size()) #torch.Size([256, 512])

                text_features = clip_model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            affinity = adapter(image_features)
            clip_logits = torch.exp(affinity @ text_features.t())
            # print("clip_logits:", clip_logits.size())
            groundtruth = torch.arange(len(images), dtype=torch.long).cuda()
            # affinity = adapter(image_features) #cache_keys torch.Size([512, 1616])
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values # cache_values torch.Size([1616, 101])
            # clip_logits = 100. * image_features @ clip_weights
            # tip_logits = clip_logits + cache_logits * alpha
            # print("tip_logits:", tip_logits.size())
            # print("cache_logits:", cache_logits.size())

            loss = F.cross_entropy(clip_logits, groundtruth)

            acc = cls_acc(clip_logits, target)
            correct_samples += acc / 100 * len(clip_logits)
            all_samples += len(clip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        clip_logits = torch.exp(affinity @ text_features.t())
        # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # clip_logits = 100. * test_features @ clip_weights
        # tip_logits = clip_logits + cache_logits * alpha

        acc = cls_acc(clip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")

    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # print("\n-------- Searching hyperparameters on the val set. --------")

    # # Search Hyperparameters
    # best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter)

    # print("\n-------- Evaluating on the test set. --------")
   
    # affinity = adapter(test_features)
    # cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    # tip_logits = clip_logits + cache_logits * best_alpha
    # acc = cls_acc(tip_logits, test_labels)
    # print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))

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
        best_acc = run_tip_adapter_F(cfg, cache_keys, cache_values, val_features, val_labels, test_features, test_labels, clip_weights, clip_model, train_loader_F, dataset.template)

        origin_acc[("origin_acc"+str(seed))] = best_acc
    
    file_path = "./output/" + str(cfg['dataset']) + '.json'
    values = list(origin_acc.values())
    mean = sum(values) / len(values)
    origin_acc["mean"] = mean
    # if not os.path.exists(file_path):
    #     os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'a',encoding='utf-8') as file:
        json.dump(origin_acc, file, indent=4, ensure_ascii=False)
           

if __name__ == '__main__':
    main()