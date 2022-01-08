import paddle
import paddle.nn as nn
from paddle.static import create_parameter

from model.vit import PatchEmbedding, TransformerBackbone, create_vision_transformer


class MaskedPatchEmbedding(nn.Layer):
    def __init__(self, in_channels, patch_size, hidden_size, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.project = nn.Linear(in_channels*patch_size**2, hidden_size)

    def forward(self, x):
        n, c, h, w = x.shape
        hn, wn = h // self.patch_size, w // self.patch_size
        x = x.reshape([n, c, hn, self.patch_size, wn, self.patch_size])
        x = x.transpose([0, 2, 4, 1, 3, 5])
        patch_num, patch_dim = hn*wn, c*self.patch_size**2
        x = x.reshape([n, patch_num, patch_dim])
        x = self.project(x)
        return x


class Decoder(nn.Layer):
    def __init__(self, layers, embedding_dim, num_heads, mlp_size, output_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder = TransformerBackbone(layers, embedding_dim, num_heads, mlp_size)
        self.project = nn.Linear(embedding_dim, output_size)

    def forward(self, x):
        x = self.decoder(x)
        x = self.project(x)
        return x


decoder_config = {
    'b2-d128': dict(
        layers=2,
        embedding_dim=128,
        num_heads=1,
        mlp_size=512,
        output_size=3*14**2
    ),
    'b2-d512': dict(
        layers=2,
        embedding_dim=512,
        num_heads=1,
        mlp_size=2048,
        output_size=3*14**2
    ),
}


def create_decoder(name):
    args = decoder_config.get(name, None)
    assert args is not None, 'invalid decoder name {}'.format(name)
    return Decoder(**args)


def random_shuffle(x_shape):
    indexes = [paddle.randperm(x_shape[1]) for _ in range(x_shape[0])]
    batch_index = paddle.stack(indexes, axis=0)
    return batch_index


def unshuffle(output, shuffle_index):
    unshuffle_index = paddle.argsort(shuffle_index, axis=1)
    x = gather(output, unshuffle_index)
    return x


def gather(x, index):
    batch_num = x.shape[0]
    batch_idx = paddle.linspace(0, batch_num - 1, batch_num).astype('int64').reshape([batch_num, 1])
    batch_idx = paddle.broadcast_to(batch_idx, [index.shape[0], index.shape[1]])
    gather_idx = paddle.stack([batch_idx, index], axis=-1)
    gather_x = paddle.gather_nd(x, gather_idx)
    return gather_x


class MAE(nn.Layer):
    def __init__(self, backbone='ViT-B', input_channel=3, input_size=224, patch_size=14, mask_ratio=0.75):
        super().__init__()
        self.input_channel = input_channel
        self.encoder = create_vision_transformer(backbone)
        self.decoder = create_decoder('b2-d512')

        self.encoder_dim = self.encoder.emdedding_dim
        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_num = (input_size // patch_size) ** 2
        self.embedding_num = int(self.patch_num * (1 - mask_ratio))
        self.patch_embedding = MaskedPatchEmbedding(
            in_channels=input_channel, patch_size=patch_size, hidden_size=self.encoder_dim)
        self.cls_token = create_parameter([1, 1, self.encoder_dim], dtype='float32')

        self.decoder_dim = self.decoder.embedding_dim
        self.en2de = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.pos_embedding = create_parameter([1, self.patch_num + 1, self.decoder_dim], dtype='float32')
        self.mask_token = create_parameter([1, 1, self.decoder_dim], dtype='float32')

    def restore_image(self, output):
        patch_width = self.input_size // self.patch_size
        x = output.reshape([output.shape[0], patch_width, patch_width,
                            self.input_channel, self.patch_size, self.patch_size])
        x = x.transpose([0, 3, 1, 4, 2, 5])
        x = x.reshape([x.shape[0], self.input_channel, self.input_size, self.input_size])
        return x

    def forward(self, x):
        x = self.patch_embedding(x)

        batch_index = random_shuffle(x.shape)
        unmasked_index = batch_index[:, :self.embedding_num]

        unmasked_embedding = gather(x, unmasked_index)
        cls_token = self.cls_token.broadcast_to([x.shape[0], 1, self.encoder_dim])
        unmasked_embedding = paddle.concat([cls_token, unmasked_embedding], axis=1)
        unmasked_embedding = self.encoder(unmasked_embedding)

        cls_embedding = unmasked_embedding[:, :1, :]
        unmasked_embedding = unmasked_embedding[:, 1:, :]

        unmasked_embedding = self.en2de(unmasked_embedding)

        masked_num = x.shape[1] - self.embedding_num
        masked_token = self.mask_token.broadcast_to([x.shape[0], masked_num, self.decoder_dim])
        full_embedding = paddle.concat([unmasked_embedding, masked_token], axis=1)

        pos_embedding = gather(self.pos_embedding, batch_index)
        full_embedding = full_embedding + pos_embedding

        img_embedding = self.decoder(full_embedding)
        img_embedding = unshuffle(img_embedding, batch_index)
        image = self.restore_image(img_embedding)

        output = {
            'image': image,
            'cls_embedding': cls_embedding,
            'unmasked_index': unmasked_index,
        }

        return output


class MAEClassifier(nn.Layer):
    def __init__(self, backbone='ViT-B', input_channel=3, input_size=224, patch_size=14, mask_ratio=0.75, num_classes=1000):
        super().__init__()
        self.mae = MAE(backbone=backbone, input_channel=input_channel,
                       input_size=input_size, patch_size=patch_size,
                       mask_ratio=mask_ratio)
        self.classifier = nn.Linear(self.mae.encoder_dim, num_classes)

    def forward(self, x):
        x = self.mae(x)
        x = self.classifier(x['cls_embedding'])

        return x


if __name__ == '__main__':
    img = paddle.rand([1, 3, 224, 224], dtype='float32')
    mae = MAE()
    out, mask1, mask2 = mae(img)
    print('output shape', out.shape)
    print('mask shape', mask1.shape, mask2.shape)
