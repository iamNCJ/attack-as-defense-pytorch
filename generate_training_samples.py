import numpy as np
import torch

from config import ATTACK_DICT, BENIGN_SAMPLE_NUM, PER_ATTACK_SAMPLE_NUM, SAMPLE_LOCATION, BS
from models import ResNet20CIFAR10
from data import CIFAR10DataModule

import foolbox as fb

from attacks import PatchedBIM
from utils.random_targets import random_targets

model = ResNet20CIFAR10()
model.load_state_dict(torch.load('./res20-cifar10.pt'))
model.eval()
dm = CIFAR10DataModule('data/cifar10/data')
data_loader = dm.get_data_loader(batch_size=BS, shuffle=False)
fmodel = fb.PyTorchModel(model, (-10, 10), device='cuda')

# judge_attack = PatchedJSMA(model, 10)
judge_attack = PatchedBIM(model)

# Adv samples
for attack_type, attack in ATTACK_DICT.items():
    counter = 0
    all_samples = []
    all_scores = []
    for images, labels in data_loader:
        images = images.to('cuda')
        labels = labels.to('cuda')
        # Generate
        predictions = fmodel(images).argmax(axis=-1)
        is_correct = predictions == labels
        raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)
        is_ok = torch.where(is_adv & is_correct)
        samples = clipped[is_ok]
        label = labels[is_ok]
        all_samples.append(samples)
        counter += samples.shape[0]
        print(f'\r {attack_type} attacks success:', counter, end="")
        if samples.shape[0] == 0:
            continue

        # Calculate attacks cost on these samples
        target = random_targets(label, 10)
        _, scores = judge_attack.perturb(samples, target)
        all_scores.append(scores)
        if counter > PER_ATTACK_SAMPLE_NUM:
            break
    all_samples = torch.cat(all_samples, dim=0)[:PER_ATTACK_SAMPLE_NUM]
    all_scores = torch.cat(all_scores, dim=0)[:PER_ATTACK_SAMPLE_NUM]
    np.save(SAMPLE_LOCATION / f'{attack_type}_samples.npy', all_samples.cpu().numpy())
    np.save(SAMPLE_LOCATION / f'{attack_type}_cost.npy', all_scores.cpu().numpy())
    print(f'saved {attack_type} samples and scores')


# Benign samples
# Generate
counter = 0
all_samples = []
all_scores = []
for images, labels in data_loader:
    images = images.to('cuda')
    labels = labels.to('cuda')
    predictions = fmodel(images).argmax(axis=-1)
    is_ok = predictions == labels
    samples = images[is_ok]
    label = labels[is_ok]
    all_samples.append(samples)
    counter += samples.shape[0]
    print(f'\r benign samples:', counter, end="")
    # Calculate attacks cost on these samples
    target = random_targets(label, 10)
    _, scores = judge_attack.perturb(samples, target)
    all_scores.append(scores)
    if counter > BENIGN_SAMPLE_NUM:
        break
all_samples = torch.cat(all_samples, dim=0)[:BENIGN_SAMPLE_NUM]
all_scores = torch.cat(all_scores, dim=0)[:BENIGN_SAMPLE_NUM]
np.save(SAMPLE_LOCATION / 'benign_samples.npy', all_samples.cpu().numpy())
np.save(SAMPLE_LOCATION / 'benign_cost.npy', all_scores.cpu().numpy())
