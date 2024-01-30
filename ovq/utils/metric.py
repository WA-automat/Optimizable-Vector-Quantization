from datasets import load_metric

bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")


def compute_metrics(pred):
    predictions, labels = pred

    # 这里假设labels是字符串列表，每个字符串表示参考答案
    references = [label for label in labels]

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)

    return {"bleu": bleu_score["score"], "rouge": rouge_score["rouge2"].mid.fmeasure}
