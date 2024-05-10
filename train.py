import numpy as np
import os, sys, yaml, importlib, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.distributions as DIS
from aim import Run, Image
import cv2 as cv
from model.model_common import mergemodel
from utils.vgg_perceptual_loss import VGGPerceptualLoss
from utils.vgg_contextual_loss import VGGContextualLoss
from utils.data_utils import *
from dataset.dataset_common import FontDataset
from tqdm import tqdm


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_obj_from_str(string):
    module, cls = string.rsplit('.', 1)
    return getattr(importlib.import_module(module, package=None), cls)


def draw(p, L, L2=None):
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
        if L[i] == 3:
            cv.line(image, (p[i, 0], p[i, 1]), (p[i+1, 0], p[i+1, 1]), (255, 0, 0), 4)
    return image


def diff(seq, device, gt=False):
    if gt:
        with torch.no_grad():
            point = seq[:, :, :2]
            mask = seq[:, :-1, 2]
            B, L, _ = point.shape
    else:
        pi, mu, sigma, rho, label = seq
        B, L, n, _ = mu.shape

        mix = DIS.Categorical(probs=pi).sample().unsqueeze(2).unsqueeze(2).repeat(1, 1, 1, 2)
        point = torch.gather(mu, 2, mix)

        mask = F.gumbel_softmax(label, tau=1, hard=True)
        mask = mask[:, :-1, 0]

    xs = torch.linspace(-1, 1, steps=128).to(device)
    ys = torch.linspace(-1, 1, steps=128).to(device)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    x = x.reshape(1, -1, 1, 1).repeat(B, 1, 1, 1)
    y = y.reshape(1, -1, 1, 1).repeat(B, 1, 1, 1)
    point = point.view(B, 1, L, 2)
    mesh = torch.cat((x, y), -1)

    a = point[:, :, :-1]
    b = point[:, :, 1:]

    L1 = torch.norm(mesh-a, p=2, dim=-1)
    L2 = torch.norm(mesh-b, p=2, dim=-1)
    L3 = (mesh-a)[:, :, :, 0] * (mesh-b)[:, :, :, 1] - (mesh-a)[:, :, :, 1] * (mesh-b)[:, :, :, 0]
    L3 = L3 / (torch.norm(a-b, p=2, dim=-1)+1e-5)
    L3 = torch.abs(L3)
    mask1 = (b-a) * (mesh-a)
    mask1 = mask1.sum(-1) < 0
    mask2 = (a-b) * (mesh-b)
    mask2 = mask2.sum(-1) < 0
    mask1 = mask1.float()
    mask2 = mask2.float()

    D = L1 * mask1 + L2 * mask2 + L3 * (1-mask1) * (1-mask2)
    mask = mask.view(B, 1, -1).repeat(1, 128*128, 1).long()
    mask = (mask == 0)

    D.masked_fill_(mask, 1e5)
    D = torch.min(D, dim=-1)[0]
    D = D.view(B, 128, 128)
    D = 1 - torch.sigmoid(100*(D-4/128.))
    return D


def mdn_sample(seq, device):
    pi, mu, sigma, rho = seq
    with torch.no_grad():
        B, L, n, _ = mu.shape
        scale_tril = torch.zeros((B, L, n, 2, 2)).to(device)
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


def example(rank, world_size, config):
    max_epoch = config['max_epoch']
    lr = config['learning_rate']
    batch_size = config['batch_size']
    imagew = config['imagew']
    seqw = config['seqw']
    plw = config['plw']
    generate_step = config['generate_step']
    show_step = config['show_step']
    accum = config['accum']
    
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if rank == 0 or rank == -1:
        name = config['name']
        if not os.path.exists('weights/'+name):
            os.mkdir('weights/'+name)

        run = Run(experiment=name)
        run['hparams'] = config

    # loss functions
    l1loss = nn.L1Loss()
    celoss = nn.CrossEntropyLoss()
    l2loss = nn.MSELoss()
    Ploss = VGGPerceptualLoss().to(rank)

    net = mergemodel(config['model']).to(rank)

    # load pretrain model
    if config['image_pretrain_model'] != '' and config['image_pretrain_model'] is not None:
        a, b = net.image_branch.load_state_dict(torch.load(config['image_pretrain_model'], map_location='cuda:{}'.format(rank)), strict=False)
    if config['seq_pretrain_model'] != '' and config['seq_pretrain_model'] is not None:
        a, b = net.seq_branch.load_state_dict(torch.load(config['seq_pretrain_model'], map_location='cuda:{}'.format(rank)), strict=False)
    if config['pretrain_model'] != '' and config['pretrain_model'] is not None:
        a, b = net.load_state_dict(torch.load(config['pretrain_model'], map_location='cuda:{}'.format(rank)), strict=False)

    net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-9)

    trainset = FontDataset(config['dataset']['train'])
    valset = FontDataset(config['dataset']['test'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, rank=rank, num_replicas=world_size, shuffle=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=8, sampler=train_sampler, pin_memory=True, drop_last=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epoch * len(trainloader))
    step = 0
    cnt = 0

    minibatch = batch_size // accum
    print(minibatch, len(trainloader))

    for epoch in range(1, max_epoch+1):
        net.train()
        train_sampler.set_epoch(epoch)
        if rank == 0 or rank == -1:
            tmploader = tqdm(trainloader)
        else:
            tmploader = trainloader
        for raw_image_source, raw_image_target, raw_image_ref, raw_seq_source, raw_seq_target, raw_seq_ref, raw_font, raw_char in tmploader:
            optim.zero_grad()
            for i in range(accum):
                image_source = raw_image_source[minibatch*i:minibatch*(i+1)].to(rank)
                image_target = raw_image_target[minibatch*i:minibatch*(i+1)].to(rank)
                image_ref = raw_image_ref[minibatch*i:minibatch*(i+1)].to(rank)
                seq_source = raw_seq_source[minibatch*i:minibatch*(i+1)].to(rank)
                seq_target = raw_seq_target[minibatch*i:minibatch*(i+1)].to(rank)
                seq_ref = raw_seq_ref[minibatch*i:minibatch*(i+1)].to(rank)
                font = raw_font[minibatch*i:minibatch*(i+1)].to(rank)
                char = raw_char[minibatch*i:minibatch*(i+1)].to(rank)

                output_image, output_seq = net((image_source, image_ref), (seq_source, seq_ref, seq_target), rank)
                # image_loss
                image_pixel_loss = l1loss(output_image['fake'], image_target[:, :1])
                image_ploss = plw * Ploss(output_image['fake'], image_target)
                image_feat_loss = l2loss(output_image['style_feat'], output_image['style_demod'])
                if config['model']['image']['dml'] == True:
                    image_dml_feat_loss = celoss(output_image['dml_feat'], font)
                    image_dml_mod_loss = celoss(output_image['dml_mod'], font)
                    image_loss = image_pixel_loss + image_ploss + image_feat_loss + image_dml_feat_loss + image_dml_mod_loss
                else:
                    image_loss = image_pixel_loss + image_ploss + image_feat_loss
                    
                # seq_loss
                seq_point_loss, seq_label_loss = mdnloss(seq_target[:, 1:], output_seq['fake'])
                seq_feat_loss = l2loss(output_seq['style_feat'], output_seq['style_demod'])

                # diff loss
                seq_skel = diff(output_seq['fake'], rank)
                seq_diff_loss = (F.relu(seq_skel-image_target[:, 0]) ** 2).mean()

                if config['model']['seq']['dml'] == True:
                    seq_dml_feat_loss = celoss(output_seq['dml_feat'], font)
                    seq_dml_mod_loss = celoss(output_seq['dml_mod'], font)
                    seq_loss = seq_point_loss + seq_label_loss + seq_feat_loss + seq_dml_feat_loss + seq_dml_mod_loss + seq_diff_loss
                else:
                    seq_loss = seq_point_loss + seq_label_loss + seq_feat_loss + seq_diff_loss
                    
                # image @ seq loss
                gt = torch.arange(len(image_source), dtype=torch.long).to(rank)
                logit = output_seq['temp'] * output_image['style_mod'] @ output_seq['style_mod'].t()
                cl_loss = celoss(logit, gt) + celoss(logit.t(), gt)
                
                loss = image_loss * imagew + seq_loss * seqw + cl_loss
                
                loss /= accum

                loss.backward()

            optim.step()
            scheduler.step()

            if step % show_step == 0 and (rank == 0 or rank == -1):
                tqdm.write('{} epoch {} step {} loss:{:.4f} image_loss:{:.4f} seq_loss:{:.4f} cl_loss:{:.4f}'.format(
                    name, epoch, step, loss.item(), image_loss.item(), seq_loss.item(), cl_loss.item()
                    ))

                run.track(loss, name='loss', step=step, context={'subset':'train'})
                run.track(image_loss, name='image_loss', step=step, context={'subset':'train'})
                run.track(seq_loss, name='seq_loss', step=step, context={'subset':'train'})
                run.track(cl_loss, name='cl_loss', step=step, context={'subset':'train'})
                run.track(scheduler.get_last_lr()[0], name='lr', step=step, context={'subset':'train'})
                
                run.track(image_pixel_loss, name='image_pixel_loss', step=step, context={'subset':'train'})
                run.track(image_ploss, name='image_ploss', step=step, context={'subset':'train'})
                run.track(image_feat_loss, name='image_feat_loss', step=step, context={'subset':'train'})
                run.track(image_dml_feat_loss, name='image_dml_feat_loss', step=step, context={'subset':'train'})
                run.track(image_dml_mod_loss, name='image_dml_mod_loss', step=step, context={'subset':'train'})
                
                run.track(seq_point_loss, name='seq_point_loss', step=step, context={'subset':'train'})
                run.track(seq_label_loss, name='seq_label_loss', step=step, context={'subset':'train'})
                run.track(seq_feat_loss, name='seq_feat_loss', step=step, context={'subset':'train'})
                run.track(seq_dml_feat_loss, name='seq_dml_feat_loss', step=step, context={'subset':'train'})
                run.track(seq_dml_mod_loss, name='seq_dml_mod_loss', step=step, context={'subset':'train'})
                run.track(seq_diff_loss, name='seq_diff_loss', step=step, context={'subset':'train'})

                run.track(Image(image_target[0]), name='image_target', step=step, context={'subset':'train'})
                run.track(Image(image_source[0]), name='image_source', step=step, context={'subset':'train'})
                run.track(Image(image_ref[0, 0]), name='image_ref', step=step, context={'subset':'train'})
                run.track(Image(output_image['fake'][0]), name='image_fake', step=step, context={'subset':'train'})

                image = draw(seq_source[0, :, :2], seq_source[0, :, 2])
                run.track(Image(image), name='seq_source', step=step, context={'subset':'train'})
                
                image = draw(seq_ref[0, 0, :, :2], seq_ref[0, 0, :, 2])
                run.track(Image(image), name='seq_ref', step=step, context={'subset':'train'})

                image = draw(seq_target[0, 1:, :2], seq_target[0, 1:, 2:].argmax(-1))
                run.track(Image(image), name='seq_real', step=step, context={'subset':'train'})

                fake = mdn_sample(output_seq['fake'][:-1], rank)
                label = output_seq['fake'][-1]
                image = draw(fake[0], label[0].argmax(-1))
                run.track(Image(image), name='seq_fake', step=step, context={'subset':'train'})
                if step % generate_step == 0:
                    with torch.no_grad():
                        _, _, output_seq, output_seq_greedy = net.module.generate((image_source, image_ref), (seq_source, seq_ref, seq_target), rank)
                        image = draw(output_seq[0, :, :2], output_seq[0, :, 2:].argmax(-1))
                        run.track(Image(image), name='seq_fake_generate', step=step, context={'subset':'train'})
                    
                        image = draw(output_seq_greedy[0, :, :2], output_seq_greedy[0, :, 2:].argmax(-1))
                        run.track(Image(image), name='seq_fake_generate_greedy', step=step, context={'subset':'train'})

            step += 1

        if (rank == 0 or rank == -1) and epoch % 10 == 0:
            torch.save(net.module.state_dict(), 'weights/{}/{}.pth.tar'.format(name, epoch))
        
            valloader = DataLoader(valset, batch_size=batch_size//accum, shuffle=False)
            net.eval()
            with torch.no_grad():
                for iter, (image_source, image_target, image_ref, seq_source, seq_target, seq_ref, font, char) in enumerate(valloader):
                    image_source = image_source.to(rank)
                    image_target = image_target.to(rank)
                    image_ref = image_ref.to(rank)
                    seq_source = seq_source.to(rank)
                    seq_target = seq_target.to(rank)
                    seq_ref = seq_ref.to(rank)
                    font = font.to(rank)
                    char = char.to(rank)
                    
                    output_image, output_seq = net((image_source, image_ref), (seq_source, seq_ref, seq_target), rank)

                    # image_loss
                    image_pixel_loss = l1loss(output_image['fake'], image_target[:, :1])
                    image_ploss = 0.01 * Ploss(output_image['fake'], image_target)
                    image_feat_loss = l2loss(output_image['style_feat'], output_image['style_demod'])
                    image_dml_feat_loss = celoss(output_image['dml_feat'], font)
                    image_dml_mod_loss = celoss(output_image['dml_mod'], font)
                    image_loss = image_pixel_loss + image_ploss + image_feat_loss# + image_dml_feat_loss + image_dml_mod_loss
                        
                    # seq_loss
                    seq_point_loss, seq_label_loss = mdnloss(seq_target[:, 1:], output_seq['fake'])
                    seq_feat_loss = l2loss(output_seq['style_feat'], output_seq['style_demod'])
                    seq_dml_feat_loss = celoss(output_seq['dml_feat'], font)
                    seq_dml_mod_loss = celoss(output_seq['dml_mod'], font)
                    seq_loss = seq_point_loss + seq_label_loss + seq_feat_loss# + seq_dml_feat_loss + seq_dml_mod_loss
                        
                    # image @ seq loss
                    gt = torch.arange(len(image_source), dtype=torch.long).to(rank)
                    logit = output_seq['temp'] * output_image['style_mod'] @ output_seq['style_mod'].t()
                    cl_loss = celoss(logit, gt) + celoss(logit.t(), gt)
                    
                    loss = image_loss + seq_loss + cl_loss
    
                    tqdm.write('val {} epoch {} iter {} loss:{:.4f} image_loss:{:.4f} seq_loss:{:.4f} cl_loss:{:.4f}'.format(
                        name, epoch, iter, loss.item(), image_loss.item(), seq_loss.item(), cl_loss.item()
                        ))

                    run.track(loss, name='loss', step=epoch, context={'subset':'val'})
                    run.track(image_loss, name='image_loss', step=epoch, context={'subset':'val'})
                    run.track(seq_loss, name='seq_loss', step=epoch, context={'subset':'val'})
                    run.track(cl_loss, name='cl_loss', step=epoch, context={'subset':'val'})
                    
                    run.track(image_pixel_loss, name='image_pixel_loss', step=epoch, context={'subset':'val'})
                    run.track(image_ploss, name='image_ploss', step=epoch, context={'subset':'val'})
                    run.track(image_feat_loss, name='image_feat_loss', step=epoch, context={'subset':'val'})
                    run.track(image_dml_feat_loss, name='image_dml_feat_loss', step=epoch, context={'subset':'val'})
                    run.track(image_dml_mod_loss, name='image_dml_mod_loss', step=epoch, context={'subset':'val'})
                    
                    run.track(seq_point_loss, name='seq_point_loss', step=epoch, context={'subset':'val'})
                    run.track(seq_label_loss, name='seq_label_loss', step=epoch, context={'subset':'val'})
                    run.track(seq_feat_loss, name='seq_feat_loss', step=epoch, context={'subset':'val'})
                    run.track(seq_dml_feat_loss, name='seq_dml_feat_loss', step=epoch, context={'subset':'val'})
                    run.track(seq_dml_mod_loss, name='seq_dml_mod_loss', step=epoch, context={'subset':'val'})

                    run.track(Image(image_target[0]), name='image_target', step=epoch, context={'subset':'val'})
                    run.track(Image(image_source[0]), name='image_source', step=epoch, context={'subset':'val'})
                    run.track(Image(image_ref[0, 0]), name='image_ref', step=epoch, context={'subset':'val'})
                    run.track(Image(output_image['fake'][0]), name='image_fake', step=epoch, context={'subset':'val'})

                    image = draw(seq_source[0, :, :2], seq_source[0, :, 2])
                    run.track(Image(image), name='seq_source', step=epoch, context={'subset':'val'})

                    image = draw(seq_target[0, 1:, :2], seq_target[0, 1:, 2:].argmax(-1))
                    run.track(Image(image), name='seq_real', step=epoch, context={'subset':'val'})

                    image = draw(seq_ref[0, 0, :, :2], seq_ref[0, 0, :, 2])
                    run.track(Image(image), name='seq_ref', step=epoch, context={'subset':'val'})
                    
                    fake = mdn_sample(output_seq['fake'][:-1], rank)
                    label = output_seq['fake'][-1]
                    image = draw(fake[0], label[0].argmax(-1))
                    run.track(Image(image), name='seq_fake', step=epoch, context={'subset':'val'})
                    
                    _, _, output_seq, output_seq_greedy = net.module.generate((image_source, image_ref), (seq_source, seq_ref, seq_target), rank)
                    image = draw(output_seq[0, :, :2], output_seq[0, :, 2:].argmax(-1))
                    run.track(Image(image), name='seq_fake_generate', step=epoch, context={'subset':'val'})
                    
                    image = draw(output_seq_greedy[0, :, :2], output_seq_greedy[0, :, 2:].argmax(-1))
                    run.track(Image(image), name='seq_fake_generate_greedy', step=epoch, context={'subset':'val'})
                    break
                
def main(config):
    world_size = torch.cuda.device_count()
    mp.spawn(example,
        args=(world_size, config),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # load config
    yaml_path = sys.argv[1]
    f = open(yaml_path, 'r', encoding='utf-8')
    config = yaml.safe_load(f.read())
    f.close()

    set_seed()
    # example('cuda:0', 1, config)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10003"
    main(config)
