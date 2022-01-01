import logging
import paddle
import os


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_logger(logfile):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def save_model(name, epoch, model, optimizer, model_dir='./train_log/model'):
    network = {
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict()
    }
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, name)
    paddle.save(network, model_path)


def load_model(model_path, model, optimizer):
    net = paddle.load(model_path)
    model.set_state_dict(net['model'])
    optimizer.set_state_dict(net['opt'])
    epoch = net['epoch']

    return epoch
