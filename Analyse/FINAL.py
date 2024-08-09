from Analyse.download_video import download_video
from Analyse.analyze import analyze_sentiment_with_distilbert
from Analyse.classify_text import classify_text
from Analyse.extract_audio import extract_audio_from_video
from Analyse.summarize_text import summarize_text
from Analyse.summarize_article import summarize_article_sentiment
from Analyse.transcribe_audio import transcribe_audio
from model.use import load_model
import zhconv
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import os
import whisper
import sys
import json
import pymongo
from reptile.douyin_reptile import dy_reptile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mongo"]
model,t5_model, tokenizer, sentiment_model, sentiment_tokenizer,classification_model, classification_tokenizer,device= load_model()
def wow(url):
    #video_url = "https://mirror.nyist.edu.cn/2.mp4"
    #outresult = dy_reptile()
    #video_url = outresult["url"]
    video_url = url
    video_path = "video.mp4"
    audio_path = "audio.wav"
    # 下载视频
    download_video(video_url, video_path)
    # 从视频中提取音频
    extract_audio_from_video(video_path, audio_path)
    # 进行语音识别
    transcription = model.transcribe(audio_path)
    #print(transcription["text"])
    # print(f"视频内容（繁体中文）:\n{transcription}")
    # 转换为简体中文
    transcription_simplified = zhconv.convert(transcription["text"], 'zh-cn')
    print(f"视频内容（简体中文）:\n{transcription_simplified}")
    # 确保在使用模型进行推断时，所有输入张量都在相同的设备上
    inputs = classification_tokenizer(transcription_simplified, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 使用多线程并行处理视频总结、情感分析和分类
    with ThreadPoolExecutor() as executor:
        future_summary = executor.submit(summarize_text, transcription_simplified, t5_model, tokenizer, device)
        future_sentiment = executor.submit(summarize_article_sentiment, transcription_simplified, device)
        future_classification = executor.submit(classify_text, transcription_simplified, classification_model,classification_tokenizer, device, 512)
        # 使用字典来保存每个任务的结果
        results = {}
        # 迭代处理每个已完成的 future
        for future in as_completed([future_summary, future_sentiment, future_classification]):
            if future == future_summary:
                results['summary'] = future.result()
            elif future == future_sentiment:
                results['sentiment_report'] = future.result()
            elif future == future_classification:
                results['classification_result'] = future.result()
    # results 字典中获取每个任务的结果
    summary = results.get('summary')
    sentiment_report = results.get('sentiment_report')
    classification_result = results.get('classification_result')

    # 清理临时文件
    os.remove(video_path)
    os.remove(audio_path)
    result = {"summary":summary,"sentiment":sentiment_report,"classification":classification_result}
    #db.dysummary.insert_one({"title":outresult["title"],"summary":summary,"sentiment":sentiment_report,"classification":classification_result,"url":video_url})
    print(result)
    return result

