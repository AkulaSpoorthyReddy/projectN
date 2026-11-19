
# STEP 39: FEATURE ENCODING AND NUMERICAL SCALING

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class FeatureEncoder:

    def __init__(self):
        self.numeric_scalers = {}
        self.categorical_encoders = {}

    def fit_transform(self, dataframe, dataset_name):
        print(f"\n========== STEP 39: FIT ENCODERS - {dataset_name} ==========")

        numeric_data = dataframe.select_dtypes(include=np.number).copy()
        categorical_data = dataframe.select_dtypes(
            include=["object", "category", "bool"]
        ).copy()

        numeric_result = pd.DataFrame(index=dataframe.index)
        categorical_result = pd.DataFrame(index=dataframe.index)

        # Fit numerical scalers on training data
        for column in numeric_data.columns:
            scaler = StandardScaler()
            values = numeric_data[[column]].replace(
                [np.inf, -np.inf], np.nan
            )
            median = values[column].median()
            values[column] = values[column].fillna(
                0 if pd.isna(median) else median
            )

            numeric_result[column] = scaler.fit_transform(values).ravel()
            self.numeric_scalers[column] = scaler

        # Fit categorical encoders on training data
        for column in categorical_data.columns:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )

            values = categorical_data[[column]].fillna("unknown")

            categorical_result[column] = encoder.fit_transform(
                values.astype(str)
            ).ravel()

            self.categorical_encoders[column] = encoder

        print("Numeric features encoded:", len(numeric_result.columns))
        print("Categorical features encoded:", len(categorical_result.columns))

        return pd.concat(
            [numeric_result, categorical_result],
            axis=1
        )

    def transform(self, dataframe):
        print("\n========== STEP 39: TRANSFORM DATA ==========")

        result = pd.DataFrame(index=dataframe.index)

        for column, scaler in self.numeric_scalers.items():
            values = dataframe[[column]].replace(
                [np.inf, -np.inf], np.nan
            )
            median = scaler.mean_[0]
            values[column] = values[column].fillna(
                median
            )

            result[column] = scaler.transform(values).ravel()

        for column, encoder in self.categorical_encoders.items():
            values = dataframe[[column]].fillna("unknown")

            result[column] = encoder.transform(
                values.astype(str)
            ).ravel()

        return result
