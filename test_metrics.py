"""Hand-computed checks independent of MMSeg; no GPU is used by these tests."""
import unittest
import numpy as np
from metrics import confusion_matrix, summarize_confusion


class MetricChecks(unittest.TestCase):
    def test_known_counts_and_ignore(self):
        gt = np.array([[0, 0, 1], [1, 255, 1]], dtype=np.uint8)
        pred = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
        matrix = confusion_matrix(gt, pred, n_classes=2)
        np.testing.assert_array_equal(matrix, [[1, 1], [1, 2]])
        result = summarize_confusion(matrix)
        self.assertEqual(result['valid_pixels'], 5)
        self.assertAlmostEqual(result['mIoU'], (1/3 + 2/4)/2*100)

    def test_absent_classes_are_excluded(self):
        result = summarize_confusion(np.array([[2, 1, 0], [0, 1, 0], [0, 0, 0]]))
        self.assertAlmostEqual(result['mIoU'], (2/3 + 1/2)/2*100)
        self.assertAlmostEqual(result['aAcc'], 75)
        self.assertEqual(result['n_classes_with_nonzero_union'], 2)
        self.assertIsNone(result['per_class_iou_percent'][2])

    def test_predicted_only_class_has_zero_iou(self):
        result = summarize_confusion(np.array([[2, 1], [0, 0]]))
        self.assertAlmostEqual(result['mIoU'], (2/3 + 0)/2*100)
        self.assertEqual(result['n_classes_with_nonzero_union'], 2)

    def test_out_of_range_predictions_are_not_ignored(self):
        with self.assertRaises(ValueError):
            confusion_matrix(np.array([[255]], dtype=np.uint8), np.array([[21]], dtype=np.uint8))

    def test_all_void_has_no_score(self):
        result = summarize_confusion(confusion_matrix(np.array([[255]], dtype=np.uint8), np.array([[0]], dtype=np.uint8)))
        self.assertIsNone(result['mIoU'])
        self.assertEqual(result['valid_pixels'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
