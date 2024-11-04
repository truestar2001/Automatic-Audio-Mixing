import sys
import os
import torch
import librosa
import numpy as np
from tqdm import tqdm
from models.AutoEncoder import AutoEncoder
from dataset.data_utils import custom_dataset
from utils import utils
import argparse
import soundfile as sf
from scipy.io.wavfile import write
os.environ["CUDA_VISIBLE_DEVICES"]="5"
def save_wav(filename, audio, sr=44100):
    """WAV 파일로 오디오 저장."""
    audio = audio*32768.0
    write(filename, sr, audio.astype(np.int16))


def load_model(checkpoint_path):
    """저장된 모델 체크포인트를 로드하는 함수."""
    model = AutoEncoder().cuda()
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])
    model.eval()  # 평가 모드로 설정
    return model

def evaluate_and_infer(model, eval_loader, criterion):
    """모델을 평가하고 MSE 값을 계산한 후, WAV 파일로 저장."""
    sessions = ['bass', 'drums', 'vocals', 'other', 'mix']
    total_loss = 0.0
    total_batches = 0
    output_dir = "inference_results_autoencoder"
    os.makedirs(output_dir, exist_ok=True)  # 결과를 저장할 디렉토리 생성

    with torch.no_grad():
        for batch_idx, (X, Y, org) in enumerate(tqdm(eval_loader, desc="Evaluating")):

            for j in range(X.size(0)):  # 배치 크기만큼 반복
                input_filedir = os.path.join(output_dir,f'{batch_idx}')
                if not os.path.exists(input_filedir):
                    os.makedirs(input_filedir)  # 디렉토리 생성
                # input_filename = os.path.join(input_filedir,'org.wav')
                # save_wav(input_filename, org.squeeze(0).cpu().numpy())
                for i, session in enumerate(sessions):
                    input_filename = os.path.join(input_filedir,f'org_{session}.wav')
                    if session == 'mix':
                        input_wav = Y[j][:,0].cpu().numpy()+Y[j][:,1].cpu().numpy()+Y[j][:,2].cpu().numpy()+Y[j][:,3].cpu().numpy()
                        save_wav(input_filename, input_wav)
                    else:
                        input_wav = Y[j][:,i].cpu().numpy()
                        save_wav(input_filename, input_wav)
                for i, session in enumerate(sessions):
                    input_filename = os.path.join(input_filedir,f'input_{session}.wav')
                    if session == 'mix':
                        input_wav = X[j][:,0].cpu().numpy()+X[j][:,1].cpu().numpy()+X[j][:,2].cpu().numpy()+X[j][:,3].cpu().numpy()
                        save_wav(input_filename, input_wav)
                    else:
                        input_wav = X[j][:,i].cpu().numpy()
                        save_wav(input_filename, input_wav)


            save_wav(input_filename, input_wav)
            X = X.permute(0, 2, 1).cuda()  # Shape 맞추기
            Y = Y.cuda()
            

            Y_hat = model(X)  # 모델 예측
            Y_hat = Y_hat.permute(0, 2, 1)  # Shape 맞추기

            # 손실 계산
            loss = criterion(Y_hat, Y)
            total_loss += loss.item()
            total_batches += 1

            # 예측 결과를 WAV로 저장
            for j in range(Y_hat.size(0)):  # 배치 크기만큼 반복
                for i, session in enumerate(sessions):
                    input_filedir = os.path.join(output_dir,f'{batch_idx}')
                    if not os.path.exists(input_filedir):
                        os.makedirs(input_filedir)  # 디렉토리 생성
                    input_filename = os.path.join(input_filedir,f'output_{session}.wav')
                    if session == 'mix':
                        input_wav = Y_hat[j][:,0].cpu().numpy()+Y_hat[j][:,1].cpu().numpy()+Y_hat[j][:,2].cpu().numpy()+Y_hat[j][:,3].cpu().numpy()
                        save_wav(input_filename, input_wav)
                    else:
                        input_wav = Y_hat[j][:,i].cpu().numpy()
                        save_wav(input_filename, input_wav)

    average_loss = total_loss / total_batches if total_batches > 0 else 0
    print(f"Average MSE Loss: {average_loss:.4f}")

def main_inference(checkpoint_path, eval_files):
    """Inference 메인 함수."""
    # 데이터셋 로드
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="/home/yoon/AudioMixing/config.json",
                        help='JSON file for configuration')
    parser.add_argument('-d', '--checkpoint_root', type=str, default="./Checkpoints/logs",
                        help='Directory for checkpoints')
    parser.add_argument('-m', '--model', type=str, default="MODEL_NAME",
                        help='Model name')
    args = parser.parse_args()
    hps = utils.get_hparams(args=args)
    eval_dataset = custom_dataset(eval_files, train=False, hparams=hps)  # hparams에 맞게 수정 필요
    eval_loader = torch.utils.data.DataLoader(eval_dataset, num_workers=4, shuffle=False, batch_size=1)

    # 모델 로드
    model = load_model(checkpoint_path)

    # 손실 함수 정의
    criterion = torch.nn.L1Loss()

    # 평가 및 추론
    evaluate_and_infer(model, eval_loader, criterion)

if __name__ == "__main__":
    checkpoint_path = "/home/yoon/AudioMixing/ckpt/AutoEncoder/step50000.pth"  # 체크포인트 경로 설정
    eval_files = "/shared/NAS_HDD/yoon/musdb_prepro/test"  # 평가 파일 경로 설정
    main_inference(checkpoint_path, eval_files)