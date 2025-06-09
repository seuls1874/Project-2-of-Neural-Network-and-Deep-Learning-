# utils/loss_tracker.py

class LossTracker:
    def __init__(self):
        self.losses_per_epoch = []

    def update(self, loss):
        self.losses_per_epoch[-1].append(loss)

    def start_new_epoch(self):
        self.losses_per_epoch.append([])

    def get_max_curve(self):
        return [max(epoch) for epoch in self.losses_per_epoch]

    def get_min_curve(self):
        return [min(epoch) for epoch in self.losses_per_epoch]

    def get_gap_curve(self):
        return [max(e) - min(e) for e in self.losses_per_epoch]
