"""Independent VOC confusion-matrix calculations; row=GT, column=prediction."""
import numpy as np


def confusion_matrix(gt, pred, n_classes=21, ignore_index=255):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.shape != pred.shape or gt.ndim != 2:
        raise ValueError('Expected matching two-dimensional label maps.')
    if not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError('Label maps must contain integers.')
    if np.any(pred < 0) or np.any(pred >= n_classes):
        raise ValueError('Prediction outside label range.')
    valid = gt != ignore_index
    if np.any(gt[valid] < 0) or np.any(gt[valid] >= n_classes):
        raise ValueError('Ground truth outside label range.')
    encoded = gt[valid].astype(np.int64) * n_classes + pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=n_classes*n_classes).reshape(n_classes, n_classes)


def summarize_confusion(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError('Expected a nonnegative square confusion matrix.')
    tp = np.diag(matrix).astype(float)
    gt_support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    union = gt_support + predicted_support - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    acc = np.divide(tp, gt_support, out=np.full_like(tp, np.nan), where=gt_support > 0)
    return {
        'aAcc': float(tp.sum()/matrix.sum()*100) if matrix.sum() else None,
        'mIoU': float(iou[union > 0].mean()*100) if np.any(union > 0) else None,
        'mAcc': float(acc[gt_support > 0].mean()*100) if np.any(gt_support > 0) else None,
        'per_class_iou_percent': [float(v*100) if np.isfinite(v) else None for v in iou],
        'gt_pixels_per_class': gt_support.tolist(),
        'prediction_pixels_per_class': predicted_support.tolist(),
        'n_classes_with_nonzero_union': int(np.count_nonzero(union)),
        'valid_pixels': int(matrix.sum()),
    }
