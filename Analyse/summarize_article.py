import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Analyse.analyze import analyze_sentiment_with_distilbert


def summarize_article_sentiment(article_text, device):
    # 加载预训练的DistilBERT模型和分词器
    model_name = "D:\moudle\lxyuandistilbert-base-multilingual-cased-sentiments-student"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    # 分析全文情感
    sentiment_scores, sentiment_class = analyze_sentiment_with_distilbert(article_text, model, tokenizer, device)

    summary_report = f"""
    情感分析报告:
    使用DistilBERT模型:
        - 总体情感分类: {sentiment_class}
        - 情感得分 (Negative, Neutral, Positive): {sentiment_scores}
    """

    return summary_report
