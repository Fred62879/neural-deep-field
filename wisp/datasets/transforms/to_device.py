

class AddToDevice:
    def __init__(self, device):
        self.device = device

    def __call__(self, inputs):
        for field in inputs.keys():
            inputs[field] = inputs[field].to(self.device)
        return inputs
