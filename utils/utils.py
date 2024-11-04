
import os
import json
import torch

def get_hparams(init=True, args=None):
  # parser = argparse.ArgumentParser()
  # parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
  #                     help='JSON file for configuration')
  # parser.add_argument('-m', '--model', type=str, required=True,
  #                     help='Model name')
  
  # args = parser.parse_args()
  # print(args)
  if args == None:
    model_checkpoint_path = os.path.join(f"./Checkpoints/logs", "MODEL_NAME")
  else:
    model_checkpoint_path = os.path.join(args.checkpoint_root, args.model)

    if not os.path.exists(model_checkpoint_path):
        os.makedirs(model_checkpoint_path)

  config_path = args.config
  config_save_path = os.path.join(model_checkpoint_path, f"{args.model}.json") 
  
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  
  hparams = HParams(**config)
  hparams.train.model_checkpoint_path = model_checkpoint_path
  hparams.train.model_name = args.model
  return hparams

def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams = HParams(**config)
  return hparams

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm



def energy_loss_torch(signal1, signal2):
    """
    Parameters:
    signal1 (torch.Tensor): 첫 번째 신호
    signal2 (torch.Tensor): 두 번째 신호
    
    Returns:
    torch.Tensor: 두 신호의 시간별 에너지 차이를 제곱하여 합한 손실 값.
    """
    # 두 신호의 절대값을 제곱하여 에너지를 계산
    energy1 = torch.abs(signal1) ** 2
    energy2 = torch.abs(signal2) ** 2
    
    # 각 시간 샘플에서 에너지 차이를 계산하고, 그 차이의 제곱을 합산하여 손실 계산
    
    
    loss = torch.sum(torch.abs((energy1 - energy2)))/energy1.size(1)
    
    return loss

def spec_l1_loss(y, y_hat, criterion):
    y_hat = spectrogram_torch(y_hat)
    y = spectrogram_torch(y)
    abs_diff = torch.abs(y) - torch.abs(y_hat)

    # L1 손실
    l1_loss = torch.abs(y - y_hat)

    # custom_loss 계산
    custom_loss = 2 * torch.abs(y - y_hat)

    # 손실값 계산
    loss = torch.mean(torch.where(abs_diff <= 0, l1_loss, custom_loss))

    return loss

def l1_loss(y, y_hat, criterion):
    y_hat = torch.abs(y_hat)
    y = torch.abs(y)
    abs_diff = torch.abs(y) - torch.abs(y_hat)

    # L1 손실
    l1_loss = torch.abs(y - y_hat)

    # custom_loss 계산
    custom_loss = 2 * torch.abs(y - y_hat)

    # 손실값 계산
    loss = torch.mean(torch.where(abs_diff <= 0, l1_loss, custom_loss))

    return loss

hann_window = {}
def spectrogram_torch(y, n_fft=2048, sampling_rate=44100, hop_size=512, win_size=2048, center=False):
    # if torch.min(y) < -1.:
    #     # print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     # print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # y shape: (batch_size, signal_length, signal_num)
    batch_size, signal_length, signal_num = y.shape

    # Reshape y to merge batch and signal_num for parallel STFT computation
    y = y.permute(0, 2, 1).reshape(batch_size * signal_num, signal_length)
    
    # Pad the signal for each channel
    y = torch.nn.functional.pad(
        y.unsqueeze(1),  # (batch_size * signal_num, 1, signal_length)
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode='reflect'
    )
    y = y.squeeze(1)  # Shape after padding: (batch_size * signal_num, signal_length + padding)

    # Perform STFT on the reshaped input
    spec = torch.stft(
        y, 
        n_fft=n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window[wnsize_dtype_device],
        center=center, 
        pad_mode='reflect', 
        normalized=False, 
        onesided=True, 
        return_complex=False
    )

    # Compute the magnitude spectrogram (sqrt of sum of squares across real and imaginary parts)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)  # Shape: (batch_size * signal_num, freq_bins, time_frames)

    # Reshape back to (batch_size, signal_num, freq_bins, time_frames)
    spec = spec.view(batch_size, signal_num, spec.shape[1], spec.shape[2])
    
    # Permute to shape (batch_size, freq_bins, time_frames, signal_num)
    spec = spec.permute(0, 2, 3, 1)  # Final shape: (batch_size, freq_bins, time_frames, signal_num)

    return spec

# Define the modified ELU function using tensors for multi-dimensional input
def shifted_elu(x, alpha=1.0):
    # Apply ELU with y-axis shift using PyTorch operations element-wise
    return torch.where(x > 0, x + 1, alpha * (torch.exp(x) - 1) + 1)