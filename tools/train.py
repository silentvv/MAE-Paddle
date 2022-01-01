import paddle
from paddle.io import DataLoader

from data.dataset import ImageNet2012, recover_normalized_image, show_image
from model.mae import MAE
from tools.misc import ensure_dir, get_logger, save_model, load_model
from argparse import ArgumentParser

import os


parser = ArgumentParser(description='Paddle MAE Training')

parser.add_argument('--model', '-m', type=str, default='ViT-B', help='the model name')
parser.add_argument('--path', '-p', type=str, help='the pretrain model path')
parser.add_argument('--model_dir', type=str, default='./train_log/models', help='the directory to save the model')

parser.add_argument('--dataset_dir', '-d', type=str, required=True, help='path to the dataset')
parser.add_argument('--batch_size', '-b', type=int, default=1)

parser.add_argument('--workers', type=int, default=1, help='number of workers')
parser.add_argument('--base_lr', type=float, default=1.5e-4)
parser.add_argument('--stop_epoch', type=int, default=1000)
parser.add_argument('--continue_train', type=bool, default=True)


def train(args):
    dataset = ImageNet2012(args.dataset_dir, 'validation')
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    model = MAE(backbone=args.model, input_size=224, patch_size=14)
    model.train()
    mse = paddle.nn.MSELoss()

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.base_lr)
    load_model('./train_log/models/latest', model, optimizer)

    ensure_dir(args.model_dir)
    logfile = os.path.join(os.path.dirname(args.model_dir), 'log.txt')
    logger = get_logger(logfile)

    try:
        for epoch in range(args.stop_epoch):
            for batch_idx, (image, target) in enumerate(data_loader()):
                out_image, unmasked_index, masked_index = model(image)
                loss = mse(image, out_image)
                optimizer.clear_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 5 == 0:
                    logger.info(f'epoch: {epoch}, batch: {batch_idx}, loss: {float(loss)}')
                if (epoch + 1) % 1 == 0:
                    ori_img = recover_normalized_image(image.numpy()[0])
                    img = recover_normalized_image(out_image.numpy()[0])
                    show_image([ori_img, img])
            if (epoch + 1) % 100 == 0:
                save_model(f'epoch_{epoch + 1}', epoch, model, optimizer, args.model_dir)
                save_model(f'latest', epoch, model, optimizer, args.model_dir)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
