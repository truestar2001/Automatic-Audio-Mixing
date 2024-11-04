#Downsampling 하는 코드.

import os
import argparse
import librosa
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.io import wavfile
from tqdm import tqdm
import random

def process(chapter, name):
    # name 's5', 'p280', 'p315' are excluded,
    for file in os.listdir(chapter):
        wav_path = os.path.join(chapter, file)
        if '.wav' in file:
            os.makedirs(os.path.join(args.out_dir1, name), exist_ok=True)
            # os.makedirs(os.path.join(args.out_dir2, speaker), exist_ok=True)
            wav, sr = librosa.load(wav_path)
            # wav, _ = librosa.effects.trim(wav, top_db=20)
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav = 0.98 * wav / peak
            wav1 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr1)
            # wav2 = librosa.resample(wav, orig_sr=sr, target_sr=args.sr2)
            # save_name = wav_name.replace("_mic2.flac", ".wav")
            save_path1 = os.path.join(args.out_dir1, name, file)
            # save_path2 = os.path.join(args.out_dir2, speaker, file)
            
            wavfile.write(
                save_path1,
                args.sr1,
                (wav1 * np.iinfo(np.int16).max).astype(np.int16)
            )
            # wavfile.write(
            #     save_path2,
            #     args.sr2,
            #     (wav2 * np.iinfo(np.int16).max).astype(np.int16)
            # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr1", type=int, default=16000, help="sampling rate")
    parser.add_argument("--in_dir", type=str, default="/home/yoon/MUSDB_converted/train", help="path to source dir")
    parser.add_argument("--out_dir1", type=str, default="/home/yoon/MUSDB_converted/preprocessed/train", help="path to target dir")
    args = parser.parse_args()


    for name in tqdm(os.listdir(args.in_dir)):
        process(os.path.join(args.in_dir, name), name)