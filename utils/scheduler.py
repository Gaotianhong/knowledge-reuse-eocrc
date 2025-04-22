from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        return [base_lr for base_lr in self.base_lrs]


class CombinedScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine_scheduler = CosineAnnealingLR(optimizer, total_steps - warmup_steps)
        super(CombinedScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            warmup_lr = WarmupScheduler(self.optimizer, self.warmup_steps, self.last_epoch).get_lr()
            return warmup_lr
        else:
            cosine_lr = self.cosine_scheduler.get_lr()
            return cosine_lr

    def step(self, epoch: int = None):
        if self.last_epoch < self.warmup_steps:
            warmup_scheduler = WarmupScheduler(self.optimizer, self.warmup_steps, self.last_epoch)
            warmup_scheduler.step(epoch)
        else:
            self.cosine_scheduler.last_epoch = self.last_epoch - self.warmup_steps
            self.cosine_scheduler.step(epoch - self.warmup_steps if epoch is not None else None)
        self.last_epoch += 1
