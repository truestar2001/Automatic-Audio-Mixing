import os
import numpy as np
import librosa
import warnings
from tqdm import tqdm


warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_sdr(folder):
    compare_list = [['output_bass', 'org_bass'], ['output_drums', 'org_drums'], ['output_vocals', 'org_vocals'], ['output_other', 'org_other'], ['output_mix', 'org_mix']]
    sdr_results = []
    for compare in compare_list:
        output_path = os.path.join(folder, compare[0] + '.wav')
        org_path = os.path.join(folder, compare[1] + '.wav')
        
        # Load audio files
        output_data, output_rate = librosa.load(output_path, sr=None)
        org_data, org_rate = librosa.load(org_path, sr=None)
        
        # Ensure both audio files have the same sampling rate and duration
        if output_rate != org_rate:
            raise ValueError(f"Sampling rates do not match for {compare[0]} and {compare[1]}")
        min_length = min(len(output_data), len(org_data))
        output_data = output_data[:min_length]
        org_data = org_data[:min_length]
        
        # Calculate SDR
        noise = org_data - output_data
        sdr = 10 * np.log10(np.sum(org_data ** 2) / np.sum(noise ** 2))
        sdr_results.append(sdr)
    return sdr_results

def calculate_sdr_random(folder):
    compare_list = [['input_bass', 'org_bass'], ['input_drums', 'org_drums'], ['input_vocals', 'org_vocals'], ['input_other', 'org_other'], ['input_mix', 'org_mix']]
    sdr_results = []
    for compare in compare_list:
        output_path = os.path.join(folder, compare[0] + '.wav')
        org_path = os.path.join(folder, compare[1] + '.wav')
        
        # Load audio files
        output_data, output_rate = librosa.load(output_path, sr=None)
        org_data, org_rate = librosa.load(org_path, sr=None)
        
        # Ensure both audio files have the same sampling rate and duration
        if output_rate != org_rate:
            raise ValueError(f"Sampling rates do not match for {compare[0]} and {compare[1]}")
        min_length = min(len(output_data), len(org_data))
        output_data = output_data[:min_length]
        org_data = org_data[:min_length]
        
        # Calculate SDR
        noise = org_data - output_data
        sdr = 10 * np.log10(np.sum(org_data ** 2) / np.sum(noise ** 2))
        sdr_results.append(sdr)
        print(sdr_results)
    return sdr_results


def main(root):
    results = []
    session_list = ['bass', 'drums', 'vocals', 'other', 'mix']
    for i in tqdm(range(45)):
        folder = os.path.join(root, str(i))
        results.append(calculate_sdr_random(folder))
    for i in range(5):
        elements = [sublist[i] for sublist in results]
        average = sum(elements) / len(elements)
        print(session_list[i], 'SDR:', average)


folder_path = '/home/yoon/AudioMixing/inference_results_timel1'
main(folder_path)
