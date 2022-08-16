from pathlib import Path
import foolbox as fb

ATTACK_DICT = {
    'fgsm': fb.attacks.FGSM(),
    'dba': fb.attacks.BoundaryAttack(),
    'bim': fb.attacks.LinfBasicIterativeAttack(),
    'df': fb.attacks.LinfDeepFoolAttack(),
}

BS = 16
SAMPLE_LOCATION = Path('./samples')
BENIGN_SAMPLE_NUM = 64
PER_ATTACK_SAMPLE_NUM = 16
