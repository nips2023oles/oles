import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from functools import partial
from model_search import MixedOp,Cell
from torch.utils.tensorboard import SummaryWriter
from operations import *


writer = SummaryWriter(log_dir="../logs")


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--cifar100', action='store_true', default=False, help='search with cifar100 dataset')
parser.add_argument('--patience', type=float, default=0.4, help='our patience')
parser.add_argument('--count', type=int, default=20, help='our count')

args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


if args.cifar100:
    CIFAR_CLASSES = 100
else:
    CIFAR_CLASSES = 10
CUR_STEP = 0


def main():

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  #checkpoint = torch.load("search-cifar100-0.3-20-20230117-082928/checkpoint.pth.tar")
  #model.load_state_dict(checkpoint['state_dict'])
  #model.set_arch_param(checkpoint['alpha'])

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  if args.cifar100:
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
  else:
    train_transform, valid_transform = utils._data_transforms_cifar10(args)  

  if args.cifar100:
      train_data = dset.CIFAR100(root=args.data,train=True, download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)
  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  best_acc=0
  best_genotype=""
  
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))
    
    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
    logging.info('train_acc %f', train_acc)

    cur_step = (epoch+1) * len(train_queue)
    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, epoch, cur_step)

    if best_acc < valid_acc:
      best_acc = valid_acc
      best_genotype = genotype
      logging.info("best genotype: %s",best_genotype)
      utils.save_checkpoint({
          'epoch': epoch + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'alpha': model.arch_parameters()
      }, False, args.save)

    logging.info('valid_acc %f, best_acc %f, ', valid_acc,best_acc)


operation_list=[Zero, AvgPool2d, MaxPool2d, Identity, SepConv3, SepConv5, DilConv3, DilConv5 ]


def freeze(m):
    logging.info(f"{m.__class__} freeze.")
    for param in m.parameters():
        param.requires_grad_(False)


def preserve_grads(m):
    if isinstance(m,Cell) or isinstance(m,MixedOp) or isinstance(m,Network):
      return 

    flag=0
    for op in operation_list:
      if isinstance(m,op):
        flag=1
        break
    
    if flag == 0:
      return

    for param in m.parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach().cpu()
            m.pre_grads.append(g)   
 


def check_grads_cosine(m):
    if isinstance(m,Cell) or isinstance(m,MixedOp) or isinstance(m,Network):
      return 

    flag = 0
    for op in operation_list:
      if isinstance(m,op):
        flag = 1
        break
    if flag == 0:
      return
    if not m.pre_grads:
      return

    i = 0
    true_i =0
    temp =0 
    
    #每个算子内部的遍历
    # print(m.__class__.__name__)
    
    for param in m.parameters():
      if param.requires_grad and param.grad is not None:
        g = param.grad.detach().cpu()
        if len(g)!=0:
          temp += torch.cosine_similarity(g, m.pre_grads[i], dim=0).mean()
          # import pdb
          # pdb.set_trace()
          true_i += 1
        i += 1
        
    if true_i != 0:
      sim_avg = temp / true_i
    m.pre_grads.clear() 
    
    m.avg += sim_avg
    m.count += 1
    #if m.avg / m.count < 0.3 and m.count : #dissimiliar
    if m.count >= 20 :
      logging.info("below 0.3 %s %f %d", m.__class__.__name__, m.avg / m.count ,m.count)        
    else: 
      m.count = 0
      m.avg=0
      
    # if nparams > 0 and diff >= nparams * args.patience:
    #     m.count += 1
    #     logging.info("over %f %s %f %d",args.patience, m.__class__.__name__, diff/nparams,m.count)
    #     if m.count >= args.count:
    #       freeze(m)
    # else:
    #     m.count = 0
    
    #writer.add_scalar('train/'+m.__class__.__name__, diff/nparams, cur_step)





def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  count = 0

  cur_step = epoch*len(train_queue)
  
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)
    
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)
    
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
   
    model.apply(partial(check_grads_cosine))
    # logging.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()  
    
    model.apply(preserve_grads)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    
    writer.add_scalar('train/loss', loss.item(), cur_step)
    cur_step += 1
    
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, epoch, cur_step):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)
    
    logits = model(input)
    loss = criterion(logits, target)
    
#     for k,v in model.named_parameters():
#         print(k,v.size())
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
   
  writer.add_scalar('val/loss', objs.avg, cur_step)
    
  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

