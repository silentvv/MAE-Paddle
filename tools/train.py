import paddle
from paddle.io import DataLoader
from paddle.optimizer.lr import LinearWarmup, CosineAnnealingDecay

from data.dataset import create_dataset, recover_normalized_image, show_image
from model.mae import MAE
from model.vit import ViT
from tools.misc import ensure_dir, get_logger, save_model, load_model
from argparse import ArgumentParser

import os


parser = ArgumentParser(description='Paddle MAE Training')

parser.add_argument('--model', '-m', type=str, default='ViT-B', help='the model name')
parser.add_argument('--model_dir', type=str, default='./train_log/models', help='the directory to save the model')
parser.add_argument('--path', '-p', type=str, default='./train_log/models/AE_latest', help='the pretrain model path')

parser.add_argument('--dataset_dir', '-d', type=str, required=True, help='path to the dataset')
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', '-b', type=int, default=1)

parser.add_argument('--workers', type=int, default=1, help='number of workers')
parser.add_argument('--base_lr', type=float, default=1e-5)
parser.add_argument('--stop_epoch', type=int, default=1000)
parser.add_argument('--continue_train', type=bool, default=True)
parser.add_argument('--train_backbone', type=bool, default=True)


def train_autoencoder(args):
    dataset = create_dataset(args.dataset_dir, 'train', input_size=args.input_size)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    ensure_dir(args.model_dir)
    logfile = os.path.join(os.path.dirname(args.model_dir), 'train.log')
    logger = get_logger(logfile)
    logger.info('=========start training MAE=========')

    model = MAE(backbone=args.model, input_size=args.input_size, patch_size=14)
    model.train()

    mse_loss = paddle.nn.MSELoss()

    cos_lr = CosineAnnealingDecay(learning_rate=args.base_lr * args.batch_size / 256, T_max=args.stop_epoch)
    lr_scheduler = LinearWarmup(cos_lr, warmup_steps=40, start_lr=1e-6, end_lr=args.base_lr * args.batch_size / 256)
    optimizer = paddle.optimizer.AdamW(
        beta1=0.9, beta2=0.95, learning_rate=lr_scheduler, weight_decay=0.05,
        parameters=model.parameters())

    latest_model_path = os.path.join(args.model_dir, 'latest')
    if args.continue_train and os.path.exists(latest_model_path):
        load_model(latest_model_path, model, optimizer)

    try:
        for epoch in range(args.stop_epoch):
            for batch_idx, (image, target) in enumerate(data_loader()):
                output = model(image)
                loss = mse_loss(image, output['image'])
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
                lr = optimizer.get_lr()
                logger.info(f'epoch: {epoch}, batch: {batch_idx}, lr: {lr} loss: {float(loss)}')
            if (epoch + 1) % 100 == 0:
                save_model(f'AE_epoch_{epoch + 1}', epoch, model, optimizer, args.model_dir)
                save_model(f'AE_latest', epoch, model, optimizer, args.model_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        exit(0)


def train_classifier(args):
    dataset = create_dataset(args.dataset_dir, 'train', input_size=args.input_size)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    ensure_dir(args.model_dir)
    logfile = os.path.join(os.path.dirname(args.model_dir), 'train.log')
    logger = get_logger(logfile)
    logger.info('=========start training classifier=========')

    mae = MAE(backbone=args.model, input_size=224, patch_size=14)
    if os.path.exists(args.path):
        logger.info('load state dict from pretrain model')
        state_dict = paddle.load(args.path)
        mae.set_state_dict(state_dict)

    model = ViT(mae, input_size=224, patch_size=14, num_classes=1000, train_backbone=args.train_backbone)
    model.train()

    ce_loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.AdamW(
        beta1=0.9, beta2=0.999, learning_rate=1e-3, weight_decay=args.weight_decay,
        parameters=model.parameters())

    try:
        for epoch in range(args.stop_epoch):
            for batch_idx, (image, target) in enumerate(data_loader()):
                logits, pred = model(image)
                target = target.astype('int64')
                loss = ce_loss(logits, target)
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
                logger.info(f'epoch: {epoch}, batch: {batch_idx}, loss: {float(loss)}')
            if (epoch + 1) % 100 == 0:
                save_model(f'epoch_{epoch + 1}', epoch, model, optimizer, args.model_dir)
                save_model(f'latest', epoch, model, optimizer, args.model_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        exit(0)


if __name__ == '__main__':
    args = parser.parse_args()
    train_autoencoder(args)
    train_classifier(args)
