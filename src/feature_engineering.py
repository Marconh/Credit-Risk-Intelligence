
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.pay_columns = [f"PAY_{i}" for i in range(1, 7)]
        self.bill_columns = [f"BILL_AMT{i}" for i in range(1, 7)]
        self.payment_columns = [f"PAY_AMT{i}" for i in range(1, 7)]

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()

        pay = X[self.pay_columns]
        bill = X[self.bill_columns]
        payments = X[self.payment_columns]

        # Historical payment delays
        X["TOTAL_DELAY_MONTHS"] = pay.clip(lower=0).sum(axis=1)
        X["MRD"] = pay.clip(lower=0).max(axis=1)
        X["EARLIER_DELAYS_N"] = (pay > 0).sum(axis=1)

        # Credit utilization
        X["UTIL_RATE_1"] = X["BILL_AMT1"] / X["LIMIT_BAL"]

        X["UTIL_RATE_AVG"] = (
            bill.div(X["LIMIT_BAL"], axis=0).mean(axis=1)
        )

        # Payment strength
        X["PAY_STRENGTH_1"] = (
            X["PAY_AMT1"] /
            X["BILL_AMT1"].where(X["BILL_AMT1"] > 0)
        )

        payment_strength = (
            payments.to_numpy() /
            bill.where(bill > 0).to_numpy()
        )

        valid_counts = np.sum(~np.isnan(payment_strength), axis=1)

        X["PAY_STRENGTH_AVG"] = np.divide(
            np.nansum(payment_strength, axis=1),
            valid_counts,
            out=np.full(len(X), np.nan),
            where=valid_counts > 0
        )

        # Billing and payment trends
        X["BILL_TREND"] = (
            bill.iloc[:, :3].mean(axis=1) -
            bill.iloc[:, 3:].mean(axis=1)
        )

        X["PAYMENT_TREND"] = (
            payments.iloc[:, :3].mean(axis=1) -
            payments.iloc[:, 3:].mean(axis=1)
        )

        # Payment behavior counts
        X["PAY_NEG2_COUNT"] = (pay == -2).sum(axis=1)
        X["PAY_NEG1_COUNT"] = (pay == -1).sum(axis=1)
        X["PAY_ZERO_COUNT"] = (pay == 0).sum(axis=1)
        X["PAY_DELAY_COUNT"] = (pay > 0).sum(axis=1)

        # Payment behavior ratios
        X["PAY_FULL_PAYMENT_RATIO"] = X["PAY_NEG1_COUNT"] / 6
        X["PAY_NO_CONSUMPTION_RATIO"] = X["PAY_NEG2_COUNT"] / 6

        # Interaction feature
        X["PAY1_BILL1_INTERACTION"] = (
            X["PAY_1"] * X["BILL_AMT1"]
        )

        # Total payment strength
        positive_bills = bill.clip(lower=0).sum(axis=1)

        X["PAY_STRENGTH_TOTAL"] = (
            payments.sum(axis=1) /
            positive_bills.where(positive_bills > 0)
        )

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_

        new_features = [
            "TOTAL_DELAY_MONTHS",
            "MRD",
            "EARLIER_DELAYS_N",
            "UTIL_RATE_1",
            "UTIL_RATE_AVG",
            "PAY_STRENGTH_1",
            "PAY_STRENGTH_AVG",
            "BILL_TREND",
            "PAYMENT_TREND",
            "PAY_NEG2_COUNT",
            "PAY_NEG1_COUNT",
            "PAY_ZERO_COUNT",
            "PAY_DELAY_COUNT",
            "PAY_FULL_PAYMENT_RATIO",
            "PAY_NO_CONSUMPTION_RATIO",
            "PAY1_BILL1_INTERACTION",
            "PAY_STRENGTH_TOTAL"
        ]

        return np.array(
            list(input_features) + new_features,
            dtype=object
        )
