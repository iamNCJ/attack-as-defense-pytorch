import numpy as np
import torch
from advertorch.attacks import JSMA
from advertorch.utils import clamp


def random_targets(labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
    """
    Given a set of correct labels, randomly changes some correct labels to target labels different from the original
    ones. These can be one-hot encoded or integers.
    :param labels: The correct labels.
    :param nb_classes: The number of classes for this model.
    :return: An array holding the randomly-selected target classes, one-hot encoded.
    """
    result = torch.zeros(labels.shape, dtype=torch.long, device=labels.device)

    for class_ind in range(nb_classes):
        other_classes = list(range(nb_classes))
        other_classes.remove(class_ind)
        in_cl = labels == class_ind
        result[in_cl] = np.random.choice(other_classes)

    return result


class PatchedJSMA(JSMA):
    """
    Patched version of JSMA that returns the final step at which the adversarial example is found.
    """
    def __init__(self, predict, num_classes,
                 clip_min=-10., clip_max=10., loss_fn=None,
                 theta=1.0, gamma=1.0, comply_cleverhans=False):
        super(PatchedJSMA, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.num_classes = num_classes
        self.theta = theta
        self.gamma = gamma
        self.comply_cleverhans = comply_cleverhans
        self.targeted = False

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        with torch.no_grad():
            outputs = self.predict(x)
        y = torch.topk(outputs, 2).indices[:, 1]
        # y = random_targets(y, self.num_classes)
        xadv = x
        batch_size = x.shape[0]
        dim_x = int(np.prod(x.shape[1:]))
        max_iters = int(dim_x * self.gamma / 2)
        search_space = x.new_ones(batch_size, dim_x).int()
        curr_step = 0
        final_steps = torch.ones(batch_size, dtype=torch.int32, device=x.device) * -1
        yadv = self._get_predicted_label(xadv)

        # Algorithm 1
        while (y != yadv).any() and curr_step < max_iters:
            grads_target, grads_other = self._compute_forward_derivative(
                xadv, y)

            # Algorithm 3
            p1, p2, valid = self._saliency_map(
                search_space, grads_target, grads_other, y)

            cond = (y != yadv) & valid

            self._update_search_space(search_space, p1, p2, cond)

            xadv = self._modify_xadv(xadv, batch_size, cond, p1, p2)
            yadv = self._get_predicted_label(xadv)

            curr_step += 1
            update_mask = (y == yadv) & (final_steps == -1)
            final_steps[update_mask == 1] = curr_step

        final_steps[y != yadv] = 2000
        xadv = clamp(xadv, min=self.clip_min, max=self.clip_max)
        return xadv, final_steps
