

class ToTensor:
    def __init__(self):
        pass

    def __call__(self, inputs):
        for field in inputs.keys():
            # print(field, type(inputs[field]).__class__.__name__)
            assert 0
            inputs[field] = torch.inputs[field].to(self.device)
        return inputs
