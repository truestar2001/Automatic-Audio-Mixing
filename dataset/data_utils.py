import sys
import os

# sys.path.append('/root/sim/VoiceConversion/FreeVC')
# sys.path.append('/root/sim/VoiceConversion/Preprocess/torch_hpss_module')
# import torch_hpss

import time
import random
import numpy as np
import torch
import torch.utils.data
torch.manual_seed(1234)
import librosa
import soundfile as sf
from scipy.io.wavfile import write
# from utils.mel_processing import spectrogram_torch, spec_to_mel_torch, mel_spectrogram_torch
# from utils.utils import load_wav_to_torch, load_filepaths_and_text, transform
#import h5py
from scipy.io.wavfile import read
# import utils.utils as utils

def load_wav_to_torch(full_path):

  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def track_dir(root_dir):
    dirs = []
    for name in os.listdir(root_dir):
        dirs.append(os.path.join(root_dir, name))
    return dirs

"""Multi speaker version"""

class custom_dataset(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths, train=True, hparams=None):
        # self.audiopaths = load_filepaths_and_text(audiopaths)
        self.audiopaths = track_dir(audiopaths)
        self.train = train
        # self.max_wav_value = hparams.data.max_wav_value #32760
        self.sampling_rate = hparams.data.sampling_rate
        self.slice_time_length = hparams.data.slice_time_length
        
        # self.filter_length  = hparams.data.filter_length
        # self.hop_length     = hparams.data.hop_length
        # self.win_length     = hparams.data.win_length
        # self.sampling_rate  = hparams.data.sampling_rate
        # self.n_mel_channels = hparams.data.n_mel_channels
        # self.mel_fmin = hparams.data.mel_fmin
        # self.mel_fmax = hparams.data.mel_fmax
        # self.use_sr = hparams.train.use_sr
        # self.use_spk = hparams.model.use_spk
        # self.spec_len = hparams.train.max_speclen
        random.seed(1234)
        # random.shuffle(self.audiopaths)
        self.source_names = ['mixture', 'bass', 'drums', 'vocals', 'other']
        self.segment_length = self.sampling_rate * self.slice_time_length 
        # self._filter()

    # def _filter(self):
    #     """
    #     Filter text & store spec lengths
    #     """
    #     # Store spectrogram lengths for Bucketing
    #     # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    #     # spec_length = wav_length // hop_length

    #     lengths = []
    #     for audiopath in self.audiopaths:
    #         lengths.append(os.path.getsize(audiopath[0]) // (2 * self.hop_length))
    #     self.lengths = lengths

    
    def get_audio(self, filename):
        audio_sources = []
        for name in self.source_names: #['mixture', 'bass', 'drums', 'vocals', 'other']
            audio_data = (load_wav_to_torch(os.path.join(filename, name+".wav"))[0][:, 0]+load_wav_to_torch(os.path.join(filename, name+".wav"))[0][:, 1])//2
            audio_sources.append(audio_data / 32768.0)
        # print(audio_sources[0].cpu().numpy())
        # print(load_wav_to_torch(os.path.join(filename, name+".wav"))[0][:, 0])
        # print(1)
        # sf.write('/home/yoon/AudioMixing/test/X.wav', audio_sources[0].cpu().numpy(), 44100)    
        # write('/home/yoon/AudioMixing/test/X.wav', 44100, (audio_sources[0].cpu().numpy()).astype(np.int16))
        mix = audio_sources[0].unsqueeze(-1) #"mixture"
        y = torch.stack(audio_sources[1:], dim=1)
        
        # Slicing
        if self.train == True:
            start_index = torch.randint(0, mix.size(0) - self.segment_length + 1, (1,))
            y = y[start_index:start_index + self.segment_length, :]
            mix = mix[start_index:start_index + self.segment_length, :]
            
            
        
        
        # Random scaling
        rand_values = torch.rand(1, 4)  # 0과 1 사이의 랜덤값을 1행 4열로 생성

        # 구간을 나누고 확률을 동일하게
        scaling_factor = torch.where(rand_values < 0.5, 1/(4-6*rand_values), 1 + (rand_values - 0.5) * 6) 
        # if self.train == False:
        #     scaling_factor = torch.tensor([[2.0, 0.5, 0.5, 2.0]])
        x = y*scaling_factor
        return x, y, mix
        # return x.unsqueeze(0), mix.unsqueeze(0)
    


            

    def __getitem__(self, index):
        return self.get_audio(self.audiopaths[index], )

    def __len__(self):
        return len(self.audiopaths)

if __name__=="__main__":
    import sys
    sys.path.append('/home/yoon/AudioMixing')
    from utils import utils
    hps = utils.get_hparams_from_file("/home/yoon/AudioMixing/config.json")
    dataset = custom_dataset("/home/yoon/MUSDB_converted/train", train=True, hparams=hps)
    a, b = dataset.__getitem__(0)
    print('end')




