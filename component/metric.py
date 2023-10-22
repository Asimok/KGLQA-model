import evaluate
import numpy as np
import transformers


# def compute_metrics(self, p: transformers.EvalPrediction):
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     preds = np.argmax(preds, axis=-1)
#
#     if preds.ndim < 3:
#         return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
#     else:
#         label_ids = p.label_ids
#         total = 0
#         num_correct = 0
#         for idx, ex_labels in enumerate(label_ids):
#             ex_labels[ex_labels == -100] = 1
#             total += 1
#             if (ex_labels == preds[idx]).all():
#                 num_correct += 1
#         return {'accuracy': num_correct / total}

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


metric = evaluate.load("component/accuracy.py")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    # .reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    # .reshape(-1)
    # print(labels.shape)
    # true_predictions = [
    #     [p for (p, l) in zip(pred, gold_label) if l != -100]
    #     for pred, gold_label in zip(preds, labels)
    # ]
    # true_labels = [
    #     [l for (p, l) in zip(pred, gold_label) if l != -100]
    #     for pred, gold_label in zip(preds, labels)
    # ]
    # preds = np.array(true_predictions).reshape(-1)
    # labels = np.array(true_labels).reshape(-1)
    return metric.compute(predictions=preds, references=labels)