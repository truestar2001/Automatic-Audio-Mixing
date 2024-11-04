import os
from tqdm import tqdm
from pydub import AudioSegment

def convert_mp3_to_mono_wav(input_folder, output_folder, genres):
    # output 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 지정된 장르에 대해 처리
    for genre in genres:
        genre_folder_path = os.path.join(input_folder, genre)
        
        # 장르 폴더가 존재하는지 확인
        if os.path.isdir(genre_folder_path):
            # 장르 폴더 내의 mp3 파일을 변환
            for file_name in tqdm(os.listdir(genre_folder_path)):
                print(file_name)
                if file_name.endswith(".wav"):
                    # mp3 파일 경로
                    mp3_path = os.path.join(genre_folder_path, file_name)
                    # wav 파일 이름과 경로 설정
                    wav_name = f"{file_name.replace('.mp3', '.wav')}"
                    wav_path = os.path.join(output_folder, wav_name)
                    
                    # output 장르 폴더가 없으면 생성
                    # os.makedirs(os.path.join(output_folder, genre), exist_ok=True)
                    
                    # 스테레오 mp3 파일을 불러와서 모노로 변환
                    audio = AudioSegment.from_mp3(mp3_path)
                    # mono_audio = audio.set_channels(2)
                    mono_audio = audio.set_frame_rate(44100)
                    
                    # wav로 저장
                    mono_audio.export(wav_path, format="wav")
                    print(f"변환 완료: {wav_path}")

input_folder = '/shared/NAS_HDD/yoon/genres_original'    # mp3 파일이 있는 폴더 경로
output_folder = '/shared/NAS_HDD/yoon/gtzan_prepro'  # 변환된 wav 파일이 저장될 경로
genres = ['pop', 'country', 'hiphop']                # 처리할 장르 목록

convert_mp3_to_mono_wav(input_folder, output_folder, genres)
