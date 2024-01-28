from torch.distributions import Normal
from torch import nn
from torch.nn import functional as F
import torch
import math


class Prior:
    """先验分布"""
    def __init__(self, sigma1=1, sigma2=0.00001, pi=0.5):
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)
        self.pi = pi

    def log_prob(self, inputs):
        """计算对数概率并求和。由于各数值相互独立，联合分布即各分布之积，取对数后变为求和。"""
        prob1 = self.normal1.log_prob(inputs).exp() # 概率密度
        prob2 = self.normal2.log_prob(inputs).exp() # 概率密度
        return (self.pi * prob1 + (1 - self.pi) * prob2).log().sum()  # 基于式38


class VariationalPoster:
    """变分后验"""
    def __init__(self):
        self.normal = Normal(0, 1)
        self.sigma = None

    def sample(self, mu, rho):
        self.mu = mu
        self.sigma = rho.exp().log1p()
        epsilon = self.normal.sample(mu.shape).to(mu.device)                        # 算法2：第5行
        return self.mu + self.sigma * epsilon  # 式33                               # 算法2：第6行

    def log_prob(self, inputs):
        """
        正态分布的对数概率密度
        log(N(x|mu, sigma)) = -log(sqrt(2*pi)) - log(sigma) - (x-mu)^2/(2*sigma^2)
        """
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma)
                - ((inputs - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior):
        """
        贝叶斯神经网络的一层。
        Args:
            in_features:  输入维度
            out_features: 输出维度
            prior:        先验分布
            mu:           mu
            rho:          rho
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.W_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        self.b_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.b_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        self.prior = prior
        self.W_variational_post = VariationalPoster()
        self.b_variational_post = VariationalPoster()

    def sample_weight(self):
        W = self.W_variational_post.sample(self.W_mu, self.W_rho)                   # 算法2：第6行
        b = self.b_variational_post.sample(self.b_mu, self.b_rho)                   # 算法2：第6行
        return W, b

    def forward(self, inputs, train=True):
        W, b = self.sample_weight()  # 采样权值矩阵和偏差向量
        outputs = F.linear(inputs, W.to(inputs.device), b.to(inputs.device))  # Wx + b

        # 预测
        if not train:
            return outputs, 0, 0

        # 训练
        # 对数先验
        log_prior = self.prior.log_prob(W).sum() + self.prior.log_prob(b).sum()      # 算法2：第7行
        # 对数变分后验
        log_va_poster = self.W_variational_post.log_prob(W) + self.b_variational_post.log_prob(b)  # 算法2：第7行
        return outputs, log_prior, log_va_poster


class BayesMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, sigma1=1, sigma2=0.00001, pi=0.5, activate='none'):
        super().__init__()
        prior = Prior(sigma1, sigma2, pi)
        self.layers = nn.ModuleList()
        for dim in hidden_dims:
            self.layers.append(BayesLinear(in_dim, dim, prior))
            in_dim = dim
        self.layers.append(BayesLinear(in_dim, out_dim, prior))

        self.act_fn = F.tanh
        if activate == 'relu':
            self.act_fn = F.relu
        elif activate == 'sigmoid':
            self.act_fn = F.sigmoid
        self.flatten = nn.Flatten()

    def run_sample(self, inputs, train):
        if len(inputs.shape) >= 3:  # 样本是矩阵而不是向量的情况（例如图像）
            inputs = self.flatten(inputs)
        log_prior, log_va_poster = 0, 0  # 对数先验，对数变分后验
        for layer in self.layers:
            model_preds, layer_log_prior, layer_log_va_poster = layer(inputs, train)
            log_prior += layer_log_prior
            log_va_poster += layer_log_va_poster
            inputs = self.act_fn(model_preds)

        return model_preds, log_prior, log_va_poster

    def forward(self, inputs, sample_num):
        log_prior_s = 0
        log_va_poser_s = 0
        model_preds_s = []

        for _ in range(sample_num):                                             # 算法2：第4行
            model_preds, log_prior, log_va_poster = self.run_sample(inputs, self.training)
            log_prior_s += log_prior           # 对数先验
            log_va_poser_s += log_va_poster    # 对数变分后验
            model_preds_s.append(model_preds)  # 模型预测

        if not self.training:
            return model_preds_s
        else:
            return model_preds_s, log_prior_s/sample_num, log_va_poser_s/sample_num



class RegressionELBOLoss(nn.Module):
    """
    用于回归问题的损失
    """
    def __init__(self, batch_num, noise_tol=0.1):
        super().__init__()
        self.batch_num = batch_num
        self.noise_tol = noise_tol

    def forward(self, model_out, targets):
        model_preds_s, log_prior, log_va_poster = model_out  # 模型输出
        log_like_s = 0
        for model_preds in model_preds_s:                                           # 算法2：第7行第3部分
            # 回归问题中模型输入被认为以预测结果为均值的高斯分布
            dist = Normal(model_preds, self.noise_tol)
            log_like_s += dist.log_prob(targets).sum()
        return 1/self.batch_num * (log_va_poster - log_prior) - log_like_s/len(model_preds_s)# 算法2：第8行


class ClassificationELBOLoss(nn.Module):
    def __init__(self, batch_num):
        super().__init__()
        self.batch_num = batch_num

    def forward(self, model_out, targets):
        model_preds_s, log_prior, log_va_poster = model_out  # 模型输出
        neg_log_like_s = 0
        for model_preds in model_preds_s:                                           # 算法2：第7行第3部分
            # 一个样本的交叉熵就是它的期望负对数似然
            neg_log_like_s += F.cross_entropy(model_preds, targets, reduction='sum')
        return 1/self.batch_num * (log_va_poster - log_prior) + neg_log_like_s/len(model_preds_s)# 算法2：第8行
