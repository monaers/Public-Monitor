from tqdm import tqdm
from moviepy.editor import VideoFileClip
def extract_audio_from_video(video_path, audio_path):
    with tqdm(total=100, desc="提取音频") as pbar:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        pbar.update(100)
    print(f"音频已提取到 {audio_path}")