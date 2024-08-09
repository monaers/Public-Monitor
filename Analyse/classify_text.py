import torch

def classify_text(text, model, tokenizer, device, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # 将输入张量移动到设备上

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
