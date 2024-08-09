from tqdm import tqdm

def summarize_text(text, model, tokenizer, device):
    prefix = "summary big to zh: "
    src_text = prefix + text
    input_ids = tokenizer(src_text, return_tensors="pt")
    input_ids = {k: v.to(device) for k, v in input_ids.items()}  # 将输入张量移动到设备上

    with tqdm(total=1, desc="文本总结") as pbar:
        generated_tokens = model.generate(**input_ids)
        summary = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pbar.update(1)

    return summary[0]