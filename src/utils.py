import sys
from torch.utils.data.dataloader import default_collate


def get_collate(name):
    if name == "identity":
        return lambda x: x
    else:
        return default_collate


def setup_logger(filename):
    f = open(filename, 'w')
    old_write = sys.stdout.write

    def _write(_s):
        f.write(_s)
        f.flush()
        old_write(_s)

    sys.stdout.write = _write
    print('=> Successfully setup logging file `{}`'.format(filename))
