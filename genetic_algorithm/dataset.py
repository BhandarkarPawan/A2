from typing import List, Tuple

import pandas as pd
from sklearn import datasets


class Dataset:
    X: pd.DataFrame
    X_small: pd.DataFrame
    y: pd.Series

    num_features: int

    def __init__(self):
        data = datasets.load_breast_cancer(return_X_y=False, as_frame=True)
        X: pd.DataFrame = data["data"]
        y: pd.Series = data["target"]

        self.num_features = len(X.columns)
        self.X = X
        self.X_small = self.X
        self.y = y

    def select_features(self, genes: List[int]) -> pd.DataFrame:
        boolean_mask = [bool(gene) for gene in genes]
        self.X_small = self.X.loc[:, boolean_mask]

    def get_selected_feature_names(self) -> List[str]:
        return self.X_small.columns.tolist()

    def get_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X_small, self.y
