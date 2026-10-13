from torch.autograd import Variable
from torchvision.utils import save_image
import random
import config
from loss_function import *
from preprocess import *
from sampling import *
from visualization import visualization

parser = argparse.ArgumentParser()

params = config.parse_args()


def main():
    model_dir = 'model/'
    test_dir = 'validation/' + 'example/'

    test_data_loader = get_loader(image_path=test_dir,
                                  image_size=params.resolution,
                                  batch_size=params.test_size,
                                  shuffle=False,
                                  num_workers=params.num_workers,
                                  mode='test',


                                  augmentation_prob=0.)

    image_size = params.resolution
    if params.RGB is True:
        channels = 3
    else:
        channels = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if params.resolution == 256:
        stage = (1, 2, 4, 8, 16, 32)
    elif params.resolution == 128:
        stage = (1, 2, 4, 8, 16)
    else:
        stage = (1, 2, 4, 8)

    model = Unet(
        dim=32,
        channels=channels,
        dim_mults=stage
    )

    model.to(device)
    model.load_state_dict(torch.load(model_dir + 'generator.pkl'))
    model.eval()

    average_num = 100

    for average_iter in range(average_num):
        for i, (condition, target) in enumerate(test_data_loader):
            condition = Variable(condition.to(device))
            if params.DDIM is False:
                samples = sample(model, condition, image_size=image_size, batch_size=params.test_size,
                                 channels=channels)
                valid_output = torch.from_numpy(samples[-1])
            else:
                samples = ddim_sample(model, condition, image_size=image_size, batch_size=params.test_size,
                                      channels=channels)
                valid_output = torch.from_numpy(samples)
            for j in range(params.test_size):
                fake_image = valid_output[j:1 + j, :, :, :]
                save_dir = 'results/' + 'example/' + '%d/' % average_iter
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                path = save_dir + "%04d.png" % (j + (params.test_size * i))
                save_image(fake_image.data,
                           os.path.join(path), nrow=4, scale_each=True)
                print('%d images are generated.' % (j + (params.test_size * i) + 1))

    colormap = 'gnuplot'
    # jet
    # rainbow
    # turbo
    # gnuplot2
    visualization(average_num, colormap)


if __name__ == '__main__':
    main()
