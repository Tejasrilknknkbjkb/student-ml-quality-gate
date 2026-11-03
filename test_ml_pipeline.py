import unittest
import os
import json
import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_csv_file_exists(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_file_exists(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 95,
            "internal_marks": 90,
            "assignment_marks": 95,
            "previous_score": 92
        }])

        prediction = model.predict(sample)
        self.assertEqual(int(prediction[0]), 1)


if __name__ == "__main__":
    unittest.main()
