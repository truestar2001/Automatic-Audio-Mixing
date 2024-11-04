import torch
import torchaudio
from demucs import pretrained
from demucs.apply import apply_model
import os
from tqdm import tqdm

# 사전 학습된 Demucs 모델 불러오기
model = pretrained.get_model('htdemucs')

# 트랙 분리 함수 정의
def separate_tracks(input_folder, output_folder):
    # 입력 폴더 내의 모든 WAV 파일 처리
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith('.wav'):
            input_file = os.path.join(input_folder, file_name)

            # 입력 오디오 파일 로드
            wav, sr = torchaudio.load(input_file)
            wav = wav.unsqueeze(0)  # 3D 텐서로 만들어서 모델에 입력
            
            # 모델을 적용하여 트랙 분리
            sources = apply_model(model, wav, device='cuda')
            sources = sources.squeeze(0)

            # 각 파일에 대해 폴더 생성
            track_names = ['drums', 'bass', 'other', 'vocals']
            output_path = os.path.join(output_folder, file_name.replace('.wav', ''))

            os.makedirs(output_path, exist_ok=True)  # 폴더 생성

            # 각 분리된 트랙을 개별 파일로 저장
            for i, track in enumerate(sources):
                torchaudio.save(os.path.join(output_path, f'{track_names[i]}.wav'), track, sr)
            
            # 원본 믹스도 저장
            torchaudio.save(os.path.join(output_path, 'mixture.wav'), wav.squeeze(0), sr)

# 예제 사용법
input_folder = '/shared/NAS_HDD/yoon/gtzan_prepro'  # 입력 폴더 경로
output_folder = '/shared/NAS_HDD/yoon/gtzan_separated'      # 출력 폴더 경로
separate_tracks(input_folder, output_folder)

print("트랙이 성공적으로 분리되어 개별 파일로 저장되었습니다.")

