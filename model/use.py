import torch
import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
def load_model():
    global model,t5_model, tokenizer, sentiment_model, sentiment_tokenizer,classification_model, classification_tokenizer
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    # 加载模型
    model_path = "D:\moudle\whisper-20231117\whisper-20231117\large-v3.pt"
    model = whisper.load_model(model_path)
    model.to(device)
    # 加载 T5 模型和分词器
    model_name = "D:\moudle\T5"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
    t5_model.to(device)
    # 使用 DistilBERT 进行情感分析的模型和分词器
    sentiment_model_name = "D:\moudle\lxyuandistilbert-base-multilingual-cased-sentiments-student"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_model.to(device)
    # 加载预训练的分类模型和分词器
    classification_model_name = "D:\moudle\Base-finetuned-jd-full-chinese"  # 替换为中文分类模型
    classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
    classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)
    classification_model.to(device)
    return model,t5_model, tokenizer, sentiment_model, sentiment_tokenizer,classification_model, classification_tokenizer,device

