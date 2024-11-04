import os
from tqdm import tqdm
from pydub import AudioSegment

def convert_stereo_to_mono(input_folder):
    # 입력 폴더 내의 모든 음악 폴더 처리
    for folder_name in tqdm(os.listdir(input_folder)):
        folder_path = os.path.join(input_folder, folder_name)

        if os.path.isdir(folder_path):  # 폴더인지 확인
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.wav'):
                    input_file_path = os.path.join(folder_path, file_name)

                    # 스테레오 WAV 파일을 불러와서 모노로 변환
                    audio = AudioSegment.from_wav(input_file_path)
                    mono_audio = audio.set_channels(1)

                    # 모노 WAV 파일로 저장 (덮어쓰기)
                    mono_audio.export(input_file_path, format="wav")
                    print(f"변환 완료: {input_file_path}")

# 예제 사용법
input_folder = '/shared/NAS_HDD/yoon/gtzan_separated'  # 음악 폴더 경로
convert_stereo_to_mono(input_folder)

print("모든 WAV 파일이 모노로 변환되었습니다.")