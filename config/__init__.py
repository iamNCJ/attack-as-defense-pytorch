from pathlib import Path
import foolbox as fb

ATTACK_DICT = {
    'fgsm': fb.attacks.FGSM(),
    'pgd': fb.attacks.PGD(),
    'bim': fb.attacks.LinfBasicIterativeAttack(),
    'df': fb.attacks.LinfDeepFoolAttack(),
}

BS = 32
SAMPLE_LOCATION = Path('./samples')
BENIGN_SAMPLE_NUM = 1000
PER_ATTACK_SAMPLE_NUM = 250
