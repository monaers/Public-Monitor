from tqdm import tqdm
def transcribe_audio(audio_path, pipe):
    with tqdm(total=100, desc="语音识别") as pbar:
        result = pipe(audio_path)
        pbar.update(100)
    return result['text']