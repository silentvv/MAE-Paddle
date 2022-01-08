import paddle.nn as nn
import paddle

from paddle.static import create_parameter


class PatchEmbedding(nn.Layer):
    def __init__(self, in_channels, patch_size, hidden_size):
        super(PatchEmbedding, self).__init__()
        self.project = nn.Conv2D(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias_attr=False)

    def forward(self, x):
        x = self.project(x)
        x = x.reshape([x.shape[0], x.shape[1], -1])
        x = x.transpose([0, 2, 1])
        return x


class PositionEmbedding(nn.Layer):
    def __init__(self, num_pos, hidden_size):
        super(PositionEmbedding, self).__init__()
        self.pos_token = create_parameter([num_pos, hidden_size], dtype='float32')

    def forward(self, x):
        pass


class MLP(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size=None, act='GELU'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if output_size is None:
            output_size = input_size
        self.fc2 = nn.Linear(hidden_size, output_size)
        if act == 'GELU':
            self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MSA(nn.Layer):
    def __init__(self, head_num, embedding_dim, hidden_size=None):
        super(MSA, self).__init__()
        self.head_num = head_num
        self.embedding_dim = embedding_dim
        if hidden_size is None:
            self.hidden_size = embedding_dim // head_num
        else:
            self.hidden_size = hidden_size
        self.qkv = nn.Linear(self.embedding_dim, 3 * self.head_num * self.hidden_size, bias_attr=False)
        self.msa = nn.Linear(self.head_num * self.hidden_size, self.embedding_dim, bias_attr=False)
        self.softmax = nn.Softmax(axis=-1)

    def _prepare_qkv(self, query):
        qkv = self.qkv(query)
        qkv = qkv.reshape([query.shape[0], query.shape[1], 3, self.head_num, self.hidden_size])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv
        return q, k, v

    def forward(self, x):
        q, k, v = self._prepare_qkv(x)
        A = paddle.matmul(q, k, transpose_y=True) / self.hidden_size**0.5
        A = self.softmax(A)
        sa = paddle.matmul(A, v)
        sa = sa.transpose([0, 2, 1, 3])
        sa = sa.reshape([sa.shape[0], sa.shape[1], -1])
        msa = self.msa(sa)
        return msa


class TransformerBlock(nn.Layer):
    def __init__(self, embedding_dim, num_heads, mlp_size):
        super(TransformerBlock, self).__init__()
        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.msa = MSA(num_heads, embedding_dim)
        self.post_norm = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_size)

    def forward(self, x):
        x = self.msa(self.pre_norm(x)) + x
        x = self.mlp(self.post_norm(x)) + x
        return x


class TransformerBackbone(nn.Layer):
    def __init__(self, layers, embedding_dim, num_heads, mlp_size):
        super(TransformerBackbone, self).__init__()
        self.emdedding_dim = embedding_dim
        self.transformer_blocks = nn.Sequential()
        for i in range(layers):
            self.transformer_blocks[str(i)] = TransformerBlock(embedding_dim, num_heads, mlp_size)

    def forward(self, x):
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return x


class ViT(nn.Layer):
    def __init__(self, mae, input_size=224, patch_size=14, num_classes=1000, train_backbone=False):
        super(ViT, self).__init__()
        self.transformer_backbone = mae.encoder
        if not train_backbone:
            self.freeze_backbone()

        self.hidden_size = self.transformer_backbone.emdedding_dim
        patch_num = (input_size // patch_size)**2
        embedding_num = patch_num + 1  # cls token

        self.patch_embedding = mae.patch_embedding
        self.cls_token = mae.cls_token
        self.pos_embedding = create_parameter([1, embedding_num, self.hidden_size], dtype='float32')
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.softmax = nn.Softmax()

    def freeze_backbone(self):
        for opr in self.transformer_backbone:
            opr.stop_gradient = True

    def forward(self, x):
        x = self.patch_embedding(x)

        cls_token = self.cls_token.broadcast_to([x.shape[0], 1, self.hidden_size])
        x = paddle.concat([x, cls_token], axis=1)

        x = x + self.pos_embedding
        x = self.transformer_backbone(x)

        x = x[:, 0]
        logits = self.classifier(x)
        pred = self.softmax(logits)

        return logits, pred


vit_config = {
    'ViT-B': dict(
        layers=12,
        embedding_dim=768,
        num_heads=12,
        mlp_size=3072
    ),
    'ViT-L': dict(
        layers=24,
        embedding_dim=1024,
        num_heads=16,
        mlp_size=4096
    ),
    'ViT-H': dict(
        layers=32,
        embedding_dim=1280,
        num_heads=16,
        mlp_size=5120
    ),
}


def create_vision_transformer(backbone):
    args = vit_config.get(backbone, None)
    assert args is not None, 'invalid backbone name: {}'.format(backbone)
    return TransformerBackbone(**args)
