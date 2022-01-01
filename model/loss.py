import paddle
import paddle.nn as nn


class MaskedMSELoss(nn.Layer):
    def forward(self, input, label, mask):
        loss = ((input - label)**2 * mask).mean()
        return loss
