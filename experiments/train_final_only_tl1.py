import sys, os
sys.path.append('/home/yoon/AudioMixing')
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# sys.path.append('/home/sim/VoiceConversion/torch_hpss')

# import torch_hpss
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim, autocast, GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import librosa
from utils import utils
import time
# 이부분 바뀜
from dataset.data_utils import (
    custom_dataset
)

from models.WaveUNet_interpolation import (
  WaveUNet
)

import wandb

torch.backends.cudnn.benchmark = True
global_step = 0
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def main():
  """Single GPU Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="/home/yoon/AudioMixing/config.json",
                      help='JSON file for configuration')
  parser.add_argument('-d', '--checkpoint_root', type=str, default="./Checkpoints/logs",
                      help='Directory for checkpoints')
  parser.add_argument('-m', '--model', type=str, default="MODEL_NAME",
                      help='Model name')
  args = parser.parse_args()
  
  hps = utils.get_hparams(args=args)

  # Single GPU setup
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = hps.train.port

  # Directly run without multi-processing
  run(0, 1, hps)


def run(rank, n_gpus, hps):
  
  global global_step
  if rank == 0 and hps.setting.log_wandb:
    wandb.init(project='AudioMixing', name=hps.model_name)
  
  # Set the current GPU
  torch.cuda.set_device(rank)

  # Set manual seed for reproducibility


  # Dataset loading
  train_dataset = custom_dataset(hps.data.training_files, train=True, hparams=hps)
  train_loader = DataLoader(train_dataset, num_workers=hps.train.num_workers, shuffle=True, batch_size=4,pin_memory=True)

  if rank == 0:
    eval_dataset = custom_dataset(hps.data.validation_files, train=True, hparams=hps)
    eval_loader = DataLoader(eval_dataset, num_workers=hps.train.num_workers, shuffle=True, batch_size=1, pin_memory=False)

  # Model initialization on the specified GPU
  net_g = WaveUNet().cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  model_path = ""  # 모델 weight 파일 경로
  global_step = 0
  try:
    current_step, optim_g = load_model_state(net_g, optim_g, model_path)
    global_step = current_step
  except:
    print("No exist model")

  
  
  criterion = torch.nn.L1Loss()

  epoch_str=1
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scaler = GradScaler(enabled=hps.train.fp16_run)

  # Training loop
  for epoch in range(epoch_str, hps.train.epochs + 1):
    # print(epoch)
    train_and_evaluate(rank, epoch, hps, net_g, optim_g, scheduler_g, scaler, [train_loader, eval_loader if rank == 0 else None], criterion)
    scheduler_g.step()

def save_model_state(model, optimizer, step, filepath):
    """모델의 상태와 현재 step을 저장하는 함수."""
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step
    }
    torch.save(state, filepath)
    print(f"Model saved to {filepath} at step {step}.")

def load_model_state(model, optimizer, filepath):
    """모델의 상태와 현재 step을 불러오는 함수."""
    try:
        state = torch.load(filepath)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        step = state['step']
        print(f"Model loaded from {filepath} at step {step}.")
        return step, optimizer
    except Exception as e:
        print(f"Error loading model state: {e}")
        return None

def evaluate(hps, generator, eval_loader, writer_eval, criterion, optimizer, best_loss=float('inf')):
    generator.eval()
    total_loss = 0.0  # Loss 값을 누적할 변수
    total_batches = 0  # 총 배치 수
    with torch.no_grad():
        for batch_idx, (X, Y, _) in enumerate(tqdm(eval_loader)):
            X = X.permute(0, 2, 1)
            X, Y = X.cuda(), Y.cuda()
            Y_hat = generator(X)
            Y_hat = Y_hat.permute(0, 2, 1)
            with autocast(device_type='cuda', enabled=False):
                loss = utils.l1_loss(Y, Y_hat, criterion)
                total_loss += loss.item() 
            total_batches += 1

    average_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"Average spec l1 loss: {average_loss:.4f}")

    # 모델 파라미터 저장
    if average_loss < best_loss:  # 평균 손실이 이전 최저 손실보다 낮으면 저장
        best_loss = average_loss
        save_model_state(generator,optimizer,global_step, os.path.join('/home/yoon/AudioMixing/ckpt/WavUNet_final_only_tl1', f'step{global_step}.pth'))  # 모델 저장 경로는 hps에서 가져온다고 가정


    generator.train()
    return best_loss  # 최저 손실 값을 반환

def train_and_evaluate(rank, epoch, hps, net_g, optim_g, scheduler_g, scaler, loaders, criterion):
    train_loader, eval_loader = loaders
    global global_step
    # running_energy_l1_loss = 0.0
    running_time_l1_loss = 0.0
    running_loss = 0.0
    net_g.train()

    best_loss = float('inf')  # 초기 최저 손실 값을 무한대로 설정

    for batch_idx, (X, Y, _) in enumerate(train_loader):
        X = X.permute(0, 2, 1)
        X, Y = X.cuda(), Y.cuda()

        with autocast(device_type='cuda', enabled=hps.train.fp16_run):
            Y_hat = net_g(X)
            Y_hat = Y_hat.permute(0, 2, 1)


        with autocast(device_type='cuda', enabled=False):
          #%
            # energy_l1_loss = utils.energy_loss_torch(Y, Y_hat)
            time_l1_loss = utils.l1_loss(Y, Y_hat, criterion)
            loss = time_l1_loss
          #%
            # running_energy_l1_loss += energy_l1_loss.item()
            running_time_l1_loss += time_l1_loss.item()
            running_loss += loss.item()
            
        optim_g.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optim_g)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0 and global_step % hps.train.log_interval == 0:
            if hps.setting.log_wandb:
                wandb.log({"loss/g_total": loss.detach().cpu().numpy()})
            if global_step % 100 == 0 and global_step != 0:
                # avg_energy_l1_loss = running_energy_l1_loss / 100
                avg_time_l1_loss = running_time_l1_loss / 100
                avg_loss = running_loss / 100
                print(f'Step {global_step}, time l1 loss: {avg_time_l1_loss},')
                # running_energy_l1_loss = 0.0
                running_time_l1_loss = 0.0
                running_loss = 0.0
            if global_step % 1000 == 0:
                best_loss = evaluate(hps, net_g, eval_loader, None, criterion,optim_g, best_loss)

        global_step += 1



if __name__ == "__main__":
  main()
