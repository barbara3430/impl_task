
class MA:
    def __init__(self, sigma):
        self.sigma = sigma
        self.archiv_params = {}

    def set(self, name, model_param):
        self.archiv_params[name] = model_param.clone()

    def set_all(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.set(name, param.data)

    def get(self, name):
        return self.archiv_params[name]

    def update(self, name, x):
        new_average = self.sigma * self.archiv_params[name] + (1 - self.sigma) * x
        self.archiv_params[name] = new_average.clone()

    def update_all(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.update(name, param.data)