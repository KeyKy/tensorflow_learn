import numpy as np

def collate_fn(data):
    # cv -> np

    imgs = [d[0][:,:,(2,1,0)] for d in data]

    labels = [d[1] for d in data]
    labels = np.stack(labels, axis=0)

    return (imgs, labels)

def add_common_args(parser):
    common = parser.add_argument_group('Common', 'the common args')
    common.add_argument('--root', type=str, help='dataset root')
    common.add_argument('--num-workers', type=int, help='number of workers in dataloader')

    return common


