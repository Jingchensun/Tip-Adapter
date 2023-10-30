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
    
    
# class TextEncoder(nn.Module):

#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#         self.cfg = cfg
#         self.classnames = classnames
#         self.clip_model = clip_model
#         self.dtype = clip_model.dtype
    
#     def forward(self):
#         temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
#         prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
#         prompts = torch.cat([clip.tokenize(p) for p in prompts])
#         prompts = prompts.to('cuda')
#         text_features = self.clip_model.encode_text(prompts)
#         x = text_features
#         return x


class CustomCLIP(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.encode_image
        self.text_encoder = clip_model.encode_text
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4).to(clip_model.dtype)

            
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        x = self.adapter(image_features)

        ratio = 0.2
        # image_features = ratio * x + (1 - ratio) * image_features
        image_features = x

        text_features = self.text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

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
    

    # clip_model = load_clip_to_cpu(cfg)
    # clip_model.float()
    model = Adapter(512, 4).to(clip_model.dtype)

    # model = CustomCLIP(clip_model)
    # for name, param in model.named_parameters():
    #     if 'adapter' not in name:
    #         param.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    cfg['train_epoch'] = 100
    cylambda1=cylambda2=0.25
    for train_idx in range(cfg['train_epoch']): #cfg['train_epoch']
        # Train
        model.train().cuda()
        correct_samples, all_samples = 0, 0
        loss_list = []
        loss_crossentopy = []
        loss_contrastive = []
        loss_inmodal = []
        loss_crossmodal = []
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
            logits_text_per_image = 100. * (affinity @ text_features.t())
            logits_image_per_text = logits_text_per_image.t()
            # print("logits_text_per_image:", logits_text_per_image) #[10.3125, 10.1406, 10.3984,  ..., 10.3125, 10.0156, 10.1406],
            # print("logits_image_per_text:", logits_image_per_text) #[10.3125, 10.1875, 10.2031,  ..., 10.2188, 10.2344, 10.1875],
            groundtruth = torch.arange(len(images), dtype=torch.long).cuda()
            # affinity = adapter(image_features) #cache_keys torch.Size([512, 1616])
            # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values # cache_values torch.Size([1616, 101])
            clip_logits2 = 10. * torch.exp(affinity @ clip_weights)
            # tip_logits = clip_logits + cache_logits * alpha
            # print("tip_logits:", tip_logits.size())
            # print("clip_logits2:", clip_logits2)

            loss1 = F.cross_entropy(logits_text_per_image, groundtruth)
            loss2 = F.cross_entropy(logits_image_per_text, groundtruth)
            loss12 = (loss1 + loss2)/2
            loss3 = F.cross_entropy(clip_logits2, target)
            # if(options.inmodal):
            #     crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
            #     inmodal_contrastive_loss = (criterion(logits_image_per_augmented_image, target) + criterion(logits_text_per_augmented_text, target)) / 2
            #     contrastive_loss = (crossmodal_contrastive_loss + inmodal_contrastive_loss) / 2
            # else:
            #     crossmodal_contrastive_loss = (criterion(logits_text_per_image, target) + criterion(logits_image_per_text, target)) / 2
            #     contrastive_loss = crossmodal_contrastive_loss
            # logits_text_per_image = umodel.logit_scale.exp() * image_embeds @ text_embeds.t()
            # logits_image_per_text = logits_text_per_image.t()
            
            inmodal_cyclic_loss = torch.tensor(0).cuda()
            # print("affinity:", affinity) #[[0.0580, 0.0000, 0.0699,  ..., 0.0483, 0.0004, 0.0854]
            # print("text_features:", text_features) # [ 0.0374,  0.0264, -0.0263,  ..., -0.0256, -0.0074,  0.0244],
            logits_image_per_image = 100. * (affinity @ affinity.t())
            # print("logits_image_per_image:", logits_image_per_image) #[[1.9082, 1.4209, 1.4189,  ..., 1.8848, 1.4014, 1.3184],
            logits_text_per_text = 100. * (text_features @ text_features.t())
            # print("logits_text_per_text:", logits_text_per_text) #[[132.2500, 109.5625,  92.3750,  ..., 132.2500, 119.6250, 112.3125],

            inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() / (100. * 100.) * len(images)
            # print("inmodal_cyclic_loss:", inmodal_cyclic_loss) #tensor(9.6328
            
            crossmodal_cyclic_loss = torch.tensor(0).cuda()
            crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() / (100. * 100.) * len(images)
            # print("crossmodal_cyclic_loss:", crossmodal_cyclic_loss) # tensor(0.2050

            cyclic_loss = cylambda1 * inmodal_cyclic_loss + cylambda2 * crossmodal_cyclic_loss
            # loss = contrastive_loss + cyclic_loss

            
            # print("loss12:", loss12) #tensor(4.6328
            # print("loss3:", loss3) #tensor(2.9863
            # print("cycli_loss:", cyclic_loss) #tensor(2.4590
            loss = loss12 + loss3 + cyclic_loss 

            tip_logits = 10. * torch.exp(affinity @ clip_weights)
            # print("tip_logits:", tip_logits)
            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())
            loss_crossentopy.append(loss12.item())
            loss_contrastive.append(loss3.item())
            loss_inmodal.append(inmodal_cyclic_loss.item())
            loss_crossmodal.append(crossmodal_cyclic_loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}),loss_crossentopy:{:.4f},loss_contrastive:{:.4f},loss_inmodal:{:.4f},loss_crossmodal:{:.4f}, Loss: {:.4f}'.format(
            current_lr, correct_samples / all_samples, correct_samples, all_samples,sum(loss_crossentopy)/len(loss_list),sum(loss_contrastive)/len(loss_list),sum(loss_inmodal)/len(loss_list),sum(loss_crossmodal)/len(loss_list), sum(loss_list)/len(loss_list)))

        # Eval
        model.eval()

        affinity = model(test_features)
        # clip_logits = torch.exp(affinity @ text_features.t())
        # cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # clip_logits = 100. * test_features @ clip_weights
        # tip_logits = clip_logits + cache_logits * alpha
        tip_logits = 10. * torch.exp(affinity @ clip_weights)

        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(model, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")

    model = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
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
    origin_acc["task"] = "loss=cyclip+crossentropy-imgfeat-100, clip weight-exp.10"
    # if not os.path.exists(file_path):
    #     os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'a',encoding='utf-8') as file:
        json.dump(origin_acc, file, indent=4, ensure_ascii=False)
           

if __name__ == '__main__':
    main()