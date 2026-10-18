import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from risk_data import load_features_and_target
from risk_inference import DATA_PATH


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitAppTests(unittest.TestCase):
    def test_individual_assessment_renders_a_prediction_and_explanation(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        self.assertEqual(len(app.exception), 0)
        overview_metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(overview_metrics["Accuracy"], "93.7%")

        app.radio[0].set_value("Student assessment")
        app.run(timeout=60)
        app.checkbox[0].check()
        next(button for button in app.button if button.label == "Assess student").click()
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("At risk" in item.value or "Not at risk" in item.value for item in app.markdown))
        self.assertEqual(len(app.get("plotly_chart")), 1)

    def test_batch_upload_scores_valid_rows_and_exposes_csv_download(self):
        X, _ = load_features_and_target(DATA_PATH)
        frame = X.iloc[:2].copy()
        frame.iloc[1, frame.columns.get_loc("StudyTimeWeekly")] = None

        app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        app.radio[0].set_value("Batch assessment")
        app.run(timeout=60)
        app.file_uploader[0].upload(
            "students.csv",
            frame.to_csv(index=False).encode("utf-8"),
            "text/csv",
        )
        app.run(timeout=60)
        next(button for button in app.button if button.label == "Validate and assess rows").click()
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Rows received"], "2")
        self.assertEqual(metrics["Rows assessed"], "1")
        self.assertEqual(metrics["Rows needing correction"], "1")
        self.assertIn("Download assessment CSV", [item.label for item in app.get("download_button")])

    def test_batch_upload_reports_missing_columns_without_crashing(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        app.radio[0].set_value("Batch assessment")
        app.run(timeout=60)
        app.file_uploader[0].upload(
            "incomplete.csv",
            b"Age,Gender\n17,0\n",
            "text/csv",
        )
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Required model columns are missing" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
