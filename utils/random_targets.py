import numpy as np
import torch


def random_targets(labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
    """
    Given a set of correct labels, randomly changes some correct labels to target labels different from the original
    ones. These can be one-hot encoded or integers.
    :param labels: The correct labels.
    :param nb_classes: The number of classes for this model.
    :return: An array holding the randomly-selected target classes, one-hot encoded.
    """
    result = torch.zeros(labels.shape, dtype=torch.long)

    for class_ind in range(nb_classes):
        other_classes = list(range(nb_classes))
        other_classes.remove(class_ind)
        in_cl = labels == class_ind
        result[in_cl] = np.random.choice(other_classes)

    return result
