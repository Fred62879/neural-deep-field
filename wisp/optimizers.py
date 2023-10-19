
from functools import partial

class multi_optimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def set_loss_func(self, criterion):
        self.criterion = criterion

    def set_current_data(self, data):
        self.data = data

    def state_dict(self):
        return [
            op.state_dict() for op in self.optimizers
        ]

    def load_state_dict(self, state_dict):
        for op, state in zip(self.optimizers, state_dict):
            op.load_state_dict(state)

    def zero_grad(self, **kwargs):
        for op in self.optimizers:
            op.zero_grad(**kwargs)

    def closure(self, op):
        # op.zero_grad()
        self.zero_grad()
        print('closure')
        loss, _ = self.criterion(self.data)
        loss.backward()
        return loss

    def step(self, closure=None):
        for op in self.optimizers:
            if op.__class__.__name__ == "LBFGS":
                op.step(closure=partial(self.closure, op))
            else:
                op.step()
