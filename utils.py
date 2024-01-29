import torch
import math
from torch.nn import functional as F
from matplotlib import pyplot as plt


def evaluate_bbb(model, val_dl, sample_num):
    """
    BBB模型的性能评价，返回正确率
    Args:
    model:      模型
    val_dl:     Dataloader
    sample_num: 每个样本采样次数
    """
    model.eval()
    total_num = 0  # 样本总数
    right_num = 0  # 预测正确的样本数量
    with torch.no_grad():
        for batch_x, batch_y in val_dl:
            preds = model(batch_x, sample_num)
            preds = torch.stack(preds, dim=-1).mean(dim=-1)  # 多次采样取均值
            right_num += (preds.argmax(dim=1) == batch_y).sum()
            total_num += len(batch_y)
    return right_num.item() / total_num


def evaluate_mcdropout(model, val_dl, sample_num):
    """
    MCDropout模型的性能评价，返回正确率
    Args:
        model:      模型
        val_dl:     Dataloader
        sample_num: 每个样本采样次数
    """
    model.eval()
    total_num = 0  # 样本总数
    right_num = 0  # 预测正确的样本数量
    with torch.no_grad():
        for batch_x, batch_y in val_dl:
            preds = [model(batch_x) for _ in range(sample_num)]
            preds = torch.stack(preds, dim=-1).mean(dim=-1)  # 多次采样取均值
            right_num += (preds.argmax(dim=1) == batch_y).sum()
            total_num += len(batch_y)
    return right_num.item() / total_num


def rotate(img, angle=30):
    """旋转图像
    Args:
        img: [channel, width, height]
        argle: 间隔旋转度数，共360/angle副图像
    """
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
    """
    Args:
        imgs: 图像列表
        rows: 行数
        gray: 是否黑白显示
        infos: 作为标题显示的信息，空值或与imgs长度相同的列表
    """
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
