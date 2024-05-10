import numpy as np
import torch


def mdnloss(real, fake, eval_mode=False):
    pi, mu, sigma, rho, fake_label = fake
    s1 = sigma[:, :, :, 0]
    s2 = sigma[:, :, :, 1]
    s1 = torch.clip(s1, 1e-6, 500.)
    s2 = torch.clip(s2, 1e-6, 500.)
    mu1 = mu[:, :, :, 0]
    mu2 = mu[:, :, :, 1]
    x1 = real[:, :, 0:1]
    x2 = real[:, :, 1:2]
    mask = 1 - real[:, :, 4]
    fake_label = fake_label.view(-1, fake_label.shape[-1])
    real_label = real[:, :, 2:].argmax(-1).view(-1).long()

    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    
    z = torch.square(norm1 / s1) + torch.square(norm2 / s2) - 2 * rho * norm1 * norm2 / s1s2
    neg_rho = torch.clip(1 - torch.square(rho), 1e-6, 1.0)

    result1 = torch.exp(-z / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    result1 = result1 / denom
    
    result1 = (result1 * pi).sum(-1)
    result1 = -torch.log(result1 + 1e-10)
    result1 = (result1 * mask).sum() / mask.sum().float()
    # result1 = (result1 * mask).mean()

    celoss = torch.nn.CrossEntropyLoss(reduction='none')
    if eval_mode:
        result2 = celoss(fake_label, real_label)
        result2 *= mask.view(-1)
        result2 = result2.mean()
    else:
        result2 = celoss(fake_label, real_label) 
        result2 = result2.mean()
    
    return result1, result2


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, sqrt_temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= sqrt_temp * sqrt_temp
    s2 *= sqrt_temp * sqrt_temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]] 
    x = np.random.multivariate_normal(mean, cov, 1) 
    return x[0][0], x[0][1]

def pad_3to5_nosos(hanzi_list, max_len):
    num = len(hanzi_list)
    result = np.zeros((num, max_len, 5), dtype=float)
    for i in range(num):
        l = len(hanzi_list[i])
        assert l <= max_len
        result[i, 0:l, 0:2] = hanzi_list[i][:, 0:2]
        result[i, 0:l, 3] = hanzi_list[i][:, 2]
        result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
        result[i, l:, 4] = 1
    return result

def pad_3to5(hanzi_list, max_len):
    """Pad the list of hanzi to be format-5. Padded hanzi shape:(max_len + 1, 5).
    input:
        - hanzi_list: list of hanzi, shape of each hanzi is (seq_len, 3)
    return:
        - result: numpy array, shape (len(hanzi_list), max_len + 1, 5)
    """
    num = len(hanzi_list)
    result = np.zeros((num, max_len + 1, 5), dtype=float)
    for i in range(num):
        l = len(hanzi_list[i])
        assert l <= max_len
        result[i, 0:l, 0:2] = hanzi_list[i][:, 0:2]
        result[i, 0:l, 3] = hanzi_list[i][:, 2]
        result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
        result[i, l:, 4] = 1
        result[i, 1:, :] = result[i, :-1, :]

        # setting S_0 (0, 0, 1, 0, 0) from paper
        result[i, 0, :] = 0
        result[i, 0, 2] = 1
        result[i, 0, 3] = 0
        result[i, 0, 4] = 0
    return result

def format3_zero_pad_to_max_len_transformer(hanzi_list, max_len):
    """
    input:
        - hanzi_list: list of hanzi, shape of each hanzi is (seq_len, 3)
    return:
        - result: numpy array, shape (len(hanzi_list), max_len+1, 3)
    """
    num = len(hanzi_list)
    result = np.zeros((num, max_len+1, 3), dtype=float)
    result[:, 0, 2] = 1
    for i in range(num):
        l = min(len(hanzi_list[i]), max_len)
        result[i, 1:l+1, :] = hanzi_list[i][:l, :]
    return result

def format3_zero_pad_to_max_len(hanzi_list, max_len):
    """
    input:
        - hanzi_list: list of hanzi, shape of each hanzi is (seq_len, 3)
    return:
        - result: numpy array, shape (len(hanzi_list), max_len, 3)
    """
    num = len(hanzi_list)
    result = np.zeros((num, max_len, 3), dtype=float)
    for i in range(num):
        l = min(len(hanzi_list[i]), max_len)
        result[i, :l, :] = hanzi_list[i][:l, :]
    return result


def random_scale(strokes, random_scale_factor):
    """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
    x_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
    y_scale_factor = (np.random.random() - 0.5) * 2 * random_scale_factor + 1.0
    result = np.copy(strokes)
    result[:, 0] *= x_scale_factor
    result[:, 1] *= y_scale_factor
    return result


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:  # current or previous point is the end of stroke
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:  # drop
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)
