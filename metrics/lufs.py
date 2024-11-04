import os
import numpy as np
import librosa
import scipy.io.wavfile as wav
import pyloudnorm as pyln
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_lufs(folder):
    compare_list = [['output_bass', 'org_bass'], ['output_drums', 'org_drums'], ['output_vocals', 'org_vocals'], ['output_other', 'org_other'], ['output_mix', 'org_mix']]
    lufs_results = []
    for compare in compare_list:
        output_path = os.path.join(folder, compare[0]+'.wav')
        org_path = os.path.join(folder, compare[1]+'.wav')
        output_rate, output_data = wav.read(output_path)
        org_rate, org_data = wav.read(org_path)
        output_data = output_data/32768
        org_data = org_data/32768
        meter = pyln.Meter(org_rate)
        iteration = org_data.shape[0]//org_rate
        output_lufs_level = meter.integrated_loudness(output_data.astype(np.float32))
        org_lufs_level = meter.integrated_loudness(org_data.astype(np.float32))
        if org_lufs_level !=0 :
            sample_lufs = abs((output_lufs_level - org_lufs_level)/org_lufs_level)
        else:
            sample_lufs = 0
        # print(output_lufs_level)
        # print(org_lufs_level)
        # print(sample_lufs)
        lufs_results.append(sample_lufs)
    return lufs_results

        
def calculate_lufs_random(folder):
    compare_list = [['input_bass', 'org_bass'], ['input_drums', 'org_drums'], ['input_vocals', 'org_vocals'], ['input_other', 'org_other'], ['input_mix', 'org_mix']]
    lufs_results = []
    for compare in compare_list:
        output_path = os.path.join(folder, compare[0]+'.wav')
        org_path = os.path.join(folder, compare[1]+'.wav')
        output_rate, output_data = wav.read(output_path)
        org_rate, org_data = wav.read(org_path)
        output_data = output_data/32768
        org_data = org_data/32768
        meter = pyln.Meter(org_rate)
        iteration = org_data.shape[0]//org_rate
        output_lufs_level = meter.integrated_loudness(output_data.astype(np.float32))
        org_lufs_level = meter.integrated_loudness(org_data.astype(np.float32))
        if org_lufs_level !=0 :
            sample_lufs = abs((output_lufs_level - org_lufs_level)/org_lufs_level)
        else:
            sample_lufs = 0
        # print(output_lufs_level)
        # print(org_lufs_level)
        # print(sample_lufs)
        lufs_results.append(sample_lufs)
    return lufs_results

def main(root):
    results = []
    session_list = ['bass', 'drums', 'vocals', 'other', 'mix']
    for i in tqdm(range(50)):
        folder = os.path.join(root, str(i))
        results.append(calculate_lufs_random(folder))
    for i in range(5):
        elements = [sublist[i] for sublist in results]
        average = sum(elements)/len(elements)
        print(session_list[i],'LUFS: ',average)
        




folder_path = '/home/yoon/AudioMixing/inference_results_timel1energy'
main(folder_path)
# calculate_lufs_shuffled(folder_path)