import unittest

from benchmark.pii_metrics import compute_pii_metrics


class PiiMetricsTests(unittest.TestCase):
    def test_masking_success_and_leakage(self):
        source = "My email is alice@example.com and phone is 123-456-7890."
        masked = "My email is [EMAIL] and phone is [PHONE]."
        privacy_mask = [
            {"value": "alice@example.com", "label": "EMAIL"},
            {"value": "123-456-7890", "label": "PHONE"},
        ]

        result = compute_pii_metrics(
            source_text=source,
            predicted_text=masked,
            target_text=masked,
            privacy_mask=privacy_mask,
        )

        self.assertAlmostEqual(result["masking_success_rate"], 1.0)
        self.assertAlmostEqual(result["pii_leakage_rate"], 0.0)
        self.assertAlmostEqual(result["text_preservation"], 1.0)

    def test_over_masking_and_leakage(self):
        source = "Call John at 555-123-4567."
        predicted = "Call [NAME] at [PHONE] and [EMAIL]."
        target = "Call [NAME] at [PHONE]."
        privacy_mask = [
            {"value": "John", "label": "PERSON"},
            {"value": "555-123-4567", "label": "PHONE"},
        ]

        result = compute_pii_metrics(
            source_text=source,
            predicted_text=predicted,
            target_text=target,
            privacy_mask=privacy_mask,
        )

        self.assertGreater(result["over_masking_rate"], 0.0)
        self.assertAlmostEqual(result["pii_leakage_rate"], 0.0)
        self.assertAlmostEqual(result["masking_success_rate"], 1.0)
        self.assertLess(result["text_preservation"], 1.0)


if __name__ == "__main__":
    unittest.main()
