import os
import requests
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import zhconv  # 使用 zhconv 进行繁简转换

# 下载视频
def download_video(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(output_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        print("下载不完整")
    else:
        print(f"视频已下载到 {output_path}")

# 从视频中提取音频
def extract_audio_from_video(video_path, audio_path):
    with tqdm(total=100, desc="提取音频") as pbar:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
        pbar.update(100)
    print(f"音频已提取到 {audio_path}")

# 使用 Whisper 进行语音识别
def transcribe_audio(audio_path, pipe):
    with tqdm(total=100, desc="语音识别") as pbar:
        result = pipe(audio_path)
        pbar.update(100)
    return result['text']

# 使用 T5 进行文本总结
def summarize_text(text, model, tokenizer):
    prefix = "summary big to zh: "
    src_text = prefix + text
    input_ids = tokenizer(src_text, return_tensors="pt")

    with tqdm(total=1, desc="文本总结") as pbar:
        generated_tokens = model.generate(**input_ids)
        summary = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pbar.update(1)

    return summary[0]

def analyze_sentiment_with_distilbert(text, model, tokenizer):
    # 对文本进行分词和编码
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测结果
    predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment_scores = predictions.tolist()[0]

    # 定义情感分类
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment_index = torch.argmax(predictions, dim=1).item()
    sentiment_class = sentiment_labels[sentiment_index]

    return sentiment_scores, sentiment_class

def summarize_article_sentiment(article_text):
    # 加载预训练的DistilBERT模型和分词器
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 分析全文情感
    sentiment_scores, sentiment_class = analyze_sentiment_with_distilbert(article_text, model, tokenizer)

    summary_report = f"""
    情感分析报告:
    使用DistilBERT模型:
        - 总体情感分类: {sentiment_class}
        - 情感得分 (Negative, Neutral, Positive): {sentiment_scores}
    """

    return summary_report

def classify_text(text, model, tokenizer, max_length=512):
    # 如果文本长度超过最大长度，则进行截断
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
    class_index = torch.argmax(predictions, dim=1).item()

    labels = ["全部", "网事小说", "突发事件", "诈骗欺诈", "历史事件", "党建活动", "产品问题", "天气预报", "违法违规", "诉讼纠纷",
            "项目规划", "人才招聘", "乡村振兴", "政策影响", "路演活动", "行业竞争", "招商引资", "客户投诉",
            "竞技比赛", "战略合作", "整改整顿", "行业研究", "召开会议", "虚假陈述", "投资融资", "宣传活动",
            "人才发展", "人事变动", "调研考察", "项目验收", "其它"]  # 替换为中文标签
    classification = labels[class_index]

    return classification

if __name__ == "__main__":

    #video_url = "https://mirror.nyist.edu.cn/2.mp4"
    video_url = "https://www.nyist.edu.cn/__local/8/34/5B/C71DD96226A3C84076A4DABEBA3_5C3C6E31_A52D959.mp4?e=.mp4"
    video_path = "video.mp4"
    audio_path = "audio.wav"

    # 下载视频
    download_video(video_url, video_path)

    # 从视频中提取音频
    extract_audio_from_video(video_path, audio_path)

    # 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # 加载 Whisper 模型
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=384,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # 使用语音识别，获取视频文字内容
    transcription = transcribe_audio(audio_path, pipe)
    # print(f"视频内容（繁体中文）:\n{transcription}")

    # 转换为简体中文
    transcription_simplified = zhconv.convert(transcription, 'zh-cn')
    print(f"视频内容（简体中文）:\n{transcription_simplified}")

    # 加载 T5 模型和分词器
    model_name = 'utrobinmv/t5_summary_en_ru_zh_base_2048'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 使用 DistilBERT 进行情感分析的模型和分词器
    sentiment_model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    # 加载预训练的分类模型和分词器
    classification_model_name = "uer/roberta-base-finetuned-jd-full-chinese"  # 替换为中文分类模型
    classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
    classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

    # 使用多线程并行处理视频总结、情感分析和分类
    with ThreadPoolExecutor() as executor:
        future_summary = executor.submit(summarize_text, transcription_simplified, t5_model, tokenizer)
        future_sentiment = executor.submit(summarize_article_sentiment, transcription_simplified)
        future_classification = executor.submit(classify_text, transcription_simplified, classification_model, classification_tokenizer)

        for future in as_completed([future_summary, future_sentiment, future_classification]):
            if future == future_summary:
                summary = future.result()
                print(f"视频总结：\n{summary}")
            elif future == future_sentiment:
                sentiment_report = future.result()
                print(sentiment_report)
            elif future == future_classification:
                classification_result = future.result()
                print(f"视频分类：\n{classification_result}")

    # 清理临时文件
    os.remove(video_path)
    os.remove(audio_path)
