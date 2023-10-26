
from functools import partial

class multi_optimizer(object):
    def __init__(self, **op):
        self.optimizers = op

    def state_dict(self):
        return [
            op.state_dict() for k, op in self.optimizers.items()
        ]

    def load_state_dict(self, state_dict):
        for (k, op), state in zip(self.optimizers.items(), state_dict):
            op.load_state_dict(state)

    def zero_grad(self, **kwargs):
        for k, op in self.optimizers.items():
            op.zero_grad(**kwargs)

    def step(self, target=None, closure=None):
        for k, op in self.optimizers.items():
            if target is not None:
                if k == target: self._step(op, closure)
            else: self._step(op, closure)

    ##########
    # Helpers
    ##########

    def _step(self, op, closure):
        if op.__class__.__name__ == "LBFGS":
            assert closure is not None
            op.step(closure=closure)
        else:
            op.step()
