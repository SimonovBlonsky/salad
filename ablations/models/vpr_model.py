import torch
import torch.nn as nn


class VPRModel(nn.Module):
    def __init__(self, encoder: nn.Module, aggregator: nn.Module, agg_arch: str):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.agg_arch = agg_arch.lower()

    def extract_features(self, x, **kwargs):
        return self.encoder(x, **kwargs)

    def aggregate(self, features):
        if self.agg_arch == "salad":
            return self.aggregator((features["feature_map"], features["global_token"]))
        return self.aggregator(features["feature_map"])

    def forward(self, x, return_features=False, **kwargs):
        features = self.extract_features(x, **kwargs)
        descriptor = self.aggregate(features)
        if not torch.isfinite(descriptor).all():
            raise ValueError("Non-finite descriptor produced by ablation VPRModel")
        if return_features:
            return descriptor, features
        return descriptor

