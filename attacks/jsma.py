import numpy as np
import torch
from advertorch.attacks import JSMA
from advertorch.utils import clamp


class PatchedJSMA(JSMA):
    """
    Patched version of JSMA that returns the final step at which the adversarial example is found.
    """
    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x
        batch_size = x.shape[0]
        dim_x = int(np.prod(x.shape[1:]))
        max_iters = int(dim_x * self.gamma / 2)
        search_space = x.new_ones(batch_size, dim_x).int()
        curr_step = 0
        final_steps = torch.ones(batch_size, dtype=torch.int32) * -1
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

        final_steps[y != yadv] = max_iters
        xadv = clamp(xadv, min=self.clip_min, max=self.clip_max)
        return xadv, final_steps
