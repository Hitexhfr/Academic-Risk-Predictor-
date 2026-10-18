import unittest

import pandas as pd

from risk_data import load_features_and_target
from risk_inference import (
    DATA_PATH,
    build_feature_specs,
    evaluate_saved_model,
    explain_prediction,
    load_model,
    model_feature_names,
    predict_batch,
    predict_one,
    validate_batch,
)


class RiskInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        cls.features = model_feature_names(cls.model)
        cls.X, cls.y = load_features_and_target(DATA_PATH)
        cls.specs = build_feature_specs(cls.X, cls.features)

    def test_saved_model_matches_single_row_dataframe_prediction(self):
        row = self.X.iloc[[0]]
        result = predict_one(self.model, row, self.features)
        expected_probability = float(self.model.predict_proba(row.loc[:, self.features])[0][1])
        self.assertEqual(result["predicted_class"], int(self.model.predict(row)[0]))
        self.assertAlmostEqual(result["risk_probability"], expected_probability)

    def test_batch_scores_valid_rows_and_retains_invalid_rows(self):
        rows = self.X.iloc[:2].copy()
        rows.iloc[1, rows.columns.get_loc("StudyTimeWeekly")] = None
        result = predict_batch(self.model, rows, self.specs)
        self.assertEqual(result["Validation status"].tolist(), ["Valid", "Invalid"])
        self.assertIn(result.loc[0, "Predicted class"], ["At risk", "Not at risk"])
        self.assertTrue(pd.isna(result.loc[1, "Predicted class"]))
        self.assertIn("StudyTimeWeekly", result.loc[1, "Validation details"])
        exported = pd.read_csv(pd.io.common.BytesIO(result.to_csv(index=False).encode("utf-8")))
        self.assertEqual(exported.columns.tolist(), result.columns.tolist())
        self.assertEqual(exported["Validation status"].tolist(), ["Valid", "Invalid"])

    def test_batch_rejects_missing_columns_and_unsupported_category_codes(self):
        with self.assertRaisesRegex(ValueError, "missing required model columns"):
            validate_batch(self.X.drop(columns=[self.features[0]]).iloc[:1], self.specs)

        row = self.X.iloc[[0]].copy()
        categorical = next(spec for spec in self.specs if spec.is_categorical)
        row.loc[row.index[0], categorical.name] = 99
        result, valid_rows = validate_batch(row, self.specs)
        self.assertEqual(result.loc[result.index[0], "Validation status"], "Invalid")
        self.assertIn("unsupported code", result.loc[result.index[0], "Validation details"])
        self.assertTrue(valid_rows.empty)

    def test_invalid_types_and_ranges_are_reported_without_crashing(self):
        row = self.X.iloc[[0]].copy()
        row["Age"] = "not-a-number"
        result, valid_rows = validate_batch(row, self.specs)
        self.assertIn("expected a numeric value", result.loc[result.index[0], "Validation details"])
        self.assertTrue(valid_rows.empty)

        row = self.X.iloc[[0]].copy()
        age_spec = next(spec for spec in self.specs if spec.name == "Age")
        row["Age"] = age_spec.maximum + 1
        result, valid_rows = validate_batch(row, self.specs)
        self.assertIn("outside the observed range", result.loc[result.index[0], "Validation details"])
        self.assertTrue(valid_rows.empty)

    def test_saved_artifact_evaluation_uses_actual_holdout(self):
        evaluation = evaluate_saved_model(self.model)
        self.assertEqual(evaluation["test_size"], 479)
        self.assertAlmostEqual(evaluation["accuracy"], 0.9373695198329853)
        self.assertAlmostEqual(evaluation["roc_auc"], 0.9876400560224089)
        self.assertEqual(evaluation["confusion_matrix"].tolist(), [[216, 8], [22, 233]])

    def test_shap_explanation_has_directions_relative_to_predicted_class(self):
        row = self.X.iloc[[0]]
        predicted = int(self.model.predict(row)[0])
        explanation = explain_prediction(self.model, row, predicted)
        self.assertEqual(set(explanation["Feature"]), set(self.features))
        self.assertTrue(
            set(explanation["Direction"]).issubset(
                {"Toward predicted class", "Away from predicted class"}
            )
        )


if __name__ == "__main__":
    unittest.main()
