import os
from pydub import AudioSegment

# 원본 폴더 경로
root_folder = "/shared/NAS_HDD/yoon/gtzan_separated"

def convert_wav_to_mp3_then_back(wav_path):
    # 임시 MP3 파일 경로
    mp3_path = wav_path.replace(".wav", ".mp3")
    
    # WAV -> MP3 변환
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    
    # MP3 -> WAV 변환 (원본 WAV 파일 덮어쓰기)
    audio_mp3 = AudioSegment.from_mp3(mp3_path)
    audio_mp3.export(wav_path, format="wav")
    
    # 임시 MP3 파일 삭제
    os.remove(mp3_path)

# 폴더 내 모든 파일 순회 및 변환 작업
for foldername in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, foldername)
    
    # 폴더 내부 파일 확인
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                wav_file_path = os.path.join(folder_path, filename)
                convert_wav_to_mp3_then_back(wav_file_path)
                print(f"Converted {wav_file_path}")

print("All files have been converted.")