import torch

def analyze_sentiment_with_distilbert(text, model, tokenizer, device):
    # 对文本进行分词和编码
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入张量移动到设备上

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
