import torch
import math
from torch.nn import functional as F
from matplotlib import pyplot as plt


def evaluate(model, val_dl, sample_num):
    """验证，输出验证集上的正确率"""
    model.eval()
    total_num = 0  # 样本总数
    right_num = 0  # 预测正确的样本数量
    for batch_x, batch_y in val_dl:
        preds = model(batch_x, sample_num)
        preds = torch.stack(preds, dim=-1).mean(dim=-1)  # 多次采样取均值
        right_num += (preds.argmax(dim=1) == batch_y).sum()
        total_num += len(batch_y)
    return right_num.item() / total_num


def rotate(img, angle=30):
    """旋转图像"""
    img = img.unsqueeze(0)
    rotation_list = range(0, 360, angle)
    image_list = []
    for r in rotation_list:
        rotation_matrix = torch.Tensor([[[math.cos(r/360.0*2*math.pi), -math.sin(r/360.0*2*math.pi), 0],
                                            [math.sin(r/360.0*2*math.pi), math.cos(r/360.0*2*math.pi), 0]]])
        grid = F.affine_grid(rotation_matrix, img.size(), align_corners=False)
        img_rotate = F.grid_sample(img, grid, align_corners=False)
        image_list.append(img_rotate)
    return torch.concat(image_list, dim=0)


def grid_show_imgs(imgs, rows=2, gray=True, infos=None):
    plt.figure(figsize=(24, 6))
    if infos is not None:
        assert len(imgs == len(infos))
    for i, img in enumerate(imgs):
        plt.subplot(rows, math.ceil(len(imgs)/rows), i+1)
        plt.axis('off')
        if infos:
            plt.gca().set_title(infos[i], size=14)
        plt.imshow(img[0, :, :].data.cpu().numpy(), cmap='gray' if gray else 'viridis')
    plt.show()


if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    import random

    data_dir = '~/datasets'

    test_ds = MNIST(data_dir, train=False, download=True, transform=ToTensor())
    img = random.choice(test_ds)

    imgs = rotate(img[0])
    grid_show_imgs(imgs, infos=[f'{i}' for i in range(len(imgs))])
