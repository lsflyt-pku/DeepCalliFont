import numpy as np
import os, sys, random, yaml, importlib
import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributions as DIS
from aim import Run, Image
import cv2 as cv
from tqdm import tqdm
from utils.data_utils import *


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def draw(p, L):
    try:
        p = p.detach().cpu().numpy()
        L = L.detach().cpu().numpy()
    except:
        pass
    p = (p + 1) / 2 * 128
    p = p.astype(np.int32)
    image = np.ones((128, 128, 3)).astype(np.uint8) * 255
    for i in range(p.shape[0]-1):
        if L[i] == 0:
            cv.line(image, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), (0, 0, 0), 4)
        if L[i] == 2:
            break
    for i in range(p.shape[0]-1):
        if L[i] == 3:
            cv.line(image, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), (0, 0, 255), 4)
        if L[i] == 2:
            break
    return image

def draw_320(p, L):
    try:
        p = p.detach().cpu().numpy()
        L = L.detach().cpu().numpy()
    except:
        pass
    p = (p + 1) / 2 * 320
    p = p.astype(np.int32)
    image = np.ones((320, 320, 3)).astype(np.uint8) * 255
    for i in range(p.shape[0]-1):
        if L[i] == 0:
            cv.line(image, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), (0, 0, 0), 8)
        if L[i] == 3:
            cv.line(image, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), (255, 0, 0), 8)
        if L[i] == 2:
            break
    return image

def mdn_sample(seq, rank, greedy=False):
    pi, mu, sigma, rho = seq
    with torch.no_grad():
        if greedy:
            return torch.gather(mu, 2, pi.argmax(-1)[:, :, None, None].repeat(1, 1, 1, 2))[:, :, 0, :]
        B, L, n, _ = mu.shape
        scale_tril = torch.zeros((B, L, n, 2, 2)).to(rank)
        s0 = sigma[:, :, :, 0]
        s1 = sigma[:, :, :, 1]
        scale_tril[:, :, :, 0, 0] = s0
        scale_tril[:, :, :, 1, 0] = rho * s1
        scale_tril[:, :, :, 1, 1] = s1 * torch.sqrt((1 - rho ** 2))
        mix = DIS.Categorical(probs=pi)
        comp = DIS.MultivariateNormal(loc=mu, scale_tril=scale_tril)
        gmm = DIS.MixtureSameFamily(mix, comp) 
        p = gmm.sample()
        return p


def example(rank, config):
    name = config['name']
    max_epoch = config['max_epoch']
    max_epoch = 80
    batch_size = config['batch_size']
    batch_size = 64

    if not os.path.exists('visual/'+name):
        os.mkdir('visual/'+name)
        os.mkdir('visual/'+name+'_all')

    net = mergemodel(config['model']).to(rank)
    a, b = net.load_state_dict(torch.load('weights/{}/{}.pth.tar'.format(name, max_epoch)))

    valset = FontDataset(config['dataset']['test'])
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)

    net.eval()
    with torch.no_grad():
        cnt = 0
        for image_source, image_target, image_ref, seq_source, seq_target, seq_ref, font, char in tqdm(valloader):
            image_source = image_source.to(rank)
            image_target = image_target.to(rank)
            image_ref = image_ref.to(rank)
            seq_source = seq_source.to(rank)
            seq_target = seq_target.to(rank)
            seq_ref = seq_ref.to(rank)
            font = font.to(rank)+1
            char = char.to(rank)+1

            image_target2 = 255 - image_target.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
            image_source2 = 255 - image_source.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
            image_ref2 = 255 - image_ref.detach().cpu().permute(0, 1, 3, 4, 2).numpy() * 255
            output_image, _, output_seq_generate, output_seq_generate_greedy = net.generate((image_source, image_ref), (seq_source, seq_ref, seq_target), rank)
            image_fake = 255 - output_image['fake'].detach().cpu().permute(0, 2, 3, 1).repeat(1, 1, 1, 3).numpy() * 255

            for i in range(font.shape[0]):
                # seq_source_image = draw(seq_source[i, :, :2], seq_source[i, :, 2])
                # seq_target_image = draw(seq_target[i, 1:, :2], seq_target[i, 1:, 2:].argmax(-1))
                # seq_fake_image = draw(output_seq_generate[i, :, :2], output_seq_generate[i, :, 2:].argmax(-1))            
                seq_fake_greedy_image = draw(output_seq_generate_greedy[i, :, :2], output_seq_generate_greedy[i, :, 2:].argmax(-1))
                # image = np.concatenate((image_source2[i], image_ref2[i, 0], image_target2[i], image_fake[i], seq_source_image, seq_target_image, seq_fake_image, seq_fake_greedy_image), 1)
                # cv.imwrite('visual/{}_all/{}_{}.png'.format(name, D[font[i].item()], char[i].item()), image)
                # seq_fake_greedy_image = draw_320(output_seq_generate_greedy[i, :, :2], output_seq_generate_greedy[i, :, 2:].argmax(-1))
                cv.imwrite('visual/{}/{}_{}_image.png'.format(name, font[i].item(), char[i].item()), image_fake[i])
                cv.imwrite('visual/{}/{}_{}_seq.png'.format(name, font[i].item(), char[i].item()), seq_fake_greedy_image)


if __name__=="__main__":
    # load config
    yaml_path = sys.argv[1]
    f = open(yaml_path, 'r', encoding='utf-8')
    config = yaml.safe_load(f.read())
    f.close()

    from model.model_common import mergemodel
    from dataset.dataset_common import FontDataset

    example(0, config)
