import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False, default='cell2sf', help='data property:'
                        )
    parser.add_argument('--batch_size', type=int, default=16, help='train batch size')
    parser.add_argument('--validation_size', type=int, default=32, help='validation batch in training')
    parser.add_argument('--test_size', type=int, default=20, help='test batch')
    parser.add_argument('--resolution', type=int, default=256, help='resolution')
    parser.add_argument('--num_epochs', type=int, default=1200, help='train epoch')
    parser.add_argument('--timesteps', type=int, default=800, help='diffusion number')
    parser.add_argument('--num_workers', type=int, default=0, help='0 for windows')
    parser.add_argument('--lrG', type=float, default=0.00015, help='learning rate for generator, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
    parser.add_argument('--lamb', type=float, default=100, help='')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
    parser.add_argument('--augmentation_prob', type=float, default=1.0, help='augmentation')
    parser.add_argument('--LV2_augmentation', type=bool, default=False, help='LV2 augmentation')
    parser.add_argument('--RGB', type=bool, default=False, help='if RGB')
    parser.add_argument('--DDIM', type=bool, default=True, help='if DDIM')
    parser.add_argument('--load_model', type=bool, default=False, help='if continual train')

    opt = parser.parse_args()

    return opt
