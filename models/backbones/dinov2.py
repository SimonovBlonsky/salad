import torch
import torch.nn as nn

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self._configure_trainable_parameters()

    def train(self, mode=True):
        if self.num_trainable_blocks <= 0:
            return super().train(False)
        return super().train(mode)

    def _configure_trainable_parameters(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if self.num_trainable_blocks <= 0:
            return

        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            for parameter in blk.parameters():
                parameter.requires_grad = True

        if self.norm_layer:
            for parameter in self.model.norm.parameters():
                parameter.requires_grad = True


    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)

        if self.num_trainable_blocks > 0:
            frozen_blocks = self.model.blocks[:-self.num_trainable_blocks]
            trainable_blocks = self.model.blocks[-self.num_trainable_blocks:]
        else:
            frozen_blocks = self.model.blocks
            trainable_blocks = []

        with torch.no_grad():
            for blk in frozen_blocks:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in trainable_blocks:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f
