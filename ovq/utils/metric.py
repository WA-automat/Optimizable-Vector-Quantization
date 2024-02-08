from datasets import load_metric

bleu_metric = load_metric("sacrebleu")


def compute_metrics(pred):
    predictions, labels = pred
    references = [label for label in labels]
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    return {"bleu": bleu_score["score"]}
