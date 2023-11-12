# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/oxford_pets.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/fgvc.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/dtd.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/eurosat.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/stanford_cars.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/sun397.yaml

# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/food101.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/caltech101.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/ucf101.yaml
# CUDA_VISIBLE_DEVICES=1 python main1.py --config configs/oxford_flowers.yaml
CUDA_VISIBLE_DEVICES=3 python main_imagenet1.py --config configs/imagenet.yaml