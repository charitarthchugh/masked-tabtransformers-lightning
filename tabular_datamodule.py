import json
import logging
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional, cast

import lightning
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

pylogger = logging.getLogger(__name__)


class CategoricalEncoder:
    """
    Encodes categorical variables into 0-indexed numeric representations for TabTransformer.

    TabTransformer expects:
    - Regular values: 0, 1, 2, ... (0-indexed per feature)
    - TabTransformer handles special tokens internally in its embedding table

    For MLM masking, we use -1 as the mask token (handled separately in forward pass).

    Attributes:
        categorical_columns (list[str]): Names of the columns to encode.
        encoders (dict[str, LabelEncoder]): Dictionary of column names mapped to their fitted
            label encoders.
    """

    def __init__(self, categorical_columns: list[str]):
        self.categorical_columns = categorical_columns
        self.encoders = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fits the LabelEncoder for each specified categorical column in the given dataframe.
        Each LabelEncoder is stored in the encoders dictionary corresponding to its column.

        Args:
            df (pd.DataFrame): The dataframe containing the categorical columns to be encoded.

        """
        df = df.map(
            lambda x: x if not pd.isna(x) else pd.NA
        ).dropna()  # Make sure the null values are of one type then
        # drop them
        for column in self.categorical_columns:
            self.encoders[column] = LabelEncoder()
            self.encoders[column].fit(df[column])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms categorical columns to 0-indexed values (0, 1, 2, ...) per feature.
        Missing values are filled with -1 and will be handled specially.

        Args:
            df (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed DataFrame with 0-indexed categorical columns

        Raises:
            ValueError: If encoder not fitted
        """
        if not self.encoders:
            raise ValueError("Encoder not fitted yet")

        df = df.copy()
        df = df.map(lambda x: x if not pd.isna(x) else pd.NA)

        for column in self.categorical_columns:
            # Map unseen values to NA
            unseen_values = ~df[column].isin(self.encoders[column].classes_)
            if unseen_values.any():
                df.loc[unseen_values, column] = pd.NA

            # Transform to 0-indexed values
            non_null_mask = df[column].notna()
            non_null_data = df.loc[non_null_mask, column]

            # Simple 0-indexed encoding (0, 1, 2, ...)
            transformed_values = self.encoders[column].transform(non_null_data)
            df.loc[non_null_mask, column] = transformed_values

        df = df.convert_dtypes()
        # Fill missing with -1 (will be handled as special case)
        df[self.categorical_columns] = df[self.categorical_columns].fillna(-1)
        return df

    def inverse_transform(self, df: pd.DataFrame):
        df = df.copy()
        for column in self.categorical_columns:
            col = df[column]
            # Mark masked/missing values
            if (col == -1).any():
                df.loc[col == -1, column] = "[MISSING/MASKED]"
            # Transform valid indices back to original values
            valid_mask = col >= 0
            if valid_mask.any():
                df.loc[valid_mask, column] = self.encoders[column].inverse_transform(
                    col.loc[valid_mask].astype(int)
                )
        return df

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    @property
    def cardinality(self) -> list[int]:
        """Returns the number of categories for each categorical column"""
        return [len(self.encoders[x].classes_) for x in self.categorical_columns]

    @staticmethod
    def from_saved_params(params: dict[str, Optional[dict]]):
        cat_encoder = CategoricalEncoder(list(params.keys()))

        encoders = {}
        for col in params:
            if params[col] is not None:
                le = LabelEncoder()
                le.set_params(**params[col])
                encoders[col] = le

        cat_encoder.encoders = encoders
        return cat_encoder

    def save_params(self):
        params = {}
        if self.encoders:
            for col in self.encoders:
                params[col] = self.encoders[col].get_params()
        else:
            params = {col: None for col in self.categorical_columns}
        return params


class TabularMetaData:
    def __init__(
        self,
        categorical_encoder: CategoricalEncoder,
        numerical_col_names: list[str] | None = None,
    ):
        self.categorical_encoder = categorical_encoder
        self.categorical_columns = categorical_encoder.categorical_columns
        self.categorical_cardinality = categorical_encoder.cardinality
        self.numerical_col_names = numerical_col_names or []

    def save(self, filepath: str | Path):
        # write categorical encoders to file
        categorical_encoder_params = self.categorical_encoder.save_params()
        filepath = Path(filepath)
        filepath.write_text(
            json.dumps(
                {
                    "categorical_encoder_params": categorical_encoder_params,
                    "numerical_col_names": self.numerical_col_names,
                }
            )
        )

    @staticmethod
    def load(filepath: str | Path) -> "TabularMetaData":
        filepath = Path(filepath)
        metadata = json.loads(filepath.read_text())
        return TabularMetaData(
            CategoricalEncoder.from_saved_params(
                metadata["categorical_encoder_params"]
            ),
            metadata["numerical_col_names"],
        )


class MaskedTabularDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[list[str]],
        numerical_columns: Optional[list[str]],
        categorical_encoder: CategoricalEncoder,
        continuous_mean_std: Optional[dict[str, dict[str, float]]] = None,
        mask_prob: float = 0.15,  # Hyperparam
        numerical_mask_type: Literal["random", "null_token", "mean"] = "null_token",
        compute_attorney_specialization=False,
    ):
        self.df = df
        self.categorical_columns = list(categorical_columns) if categorical_columns else []
        self.numerical_columns = list(numerical_columns) if numerical_columns else []
        self.categorical_encoder = categorical_encoder
        self.continuous_mean_std = continuous_mean_std
        self.mask_prob = mask_prob
        self.numerical_mask_type = cast(
            Literal["random", "null_token", "mean"], numerical_mask_type
        )
        self.compute_attorney_specialization = compute_attorney_specialization

        if self.numerical_mask_type not in ["random", "null_token", "mean"]:
            raise ValueError(
                "numerical_mask_type must be one of 'random', 'null_token', 'mean'"
            )
        if not self.continuous_mean_std and self.numerical_mask_type == "mean":
            raise ValueError(
                "continuous_mean_std must be provided if numerical_mask_type is 'mean'"
            )
        if self.categorical_columns and self.numerical_columns:
            # Reorder the columns so that the positioning is consistent
            self.df = self.df[
                list(self.categorical_columns) + list(self.numerical_columns)
            ]
            self.df = self.categorical_encoder.transform(self.df)
        elif self.categorical_columns:
            self.df = self.df[list(self.categorical_columns)]
            self.df = self.categorical_encoder.transform(self.df)
        elif self.numerical_columns:
            self.df = self.df[list(self.numerical_columns)]
        else:
            raise ValueError(
                "At least one of categorical_columns or numerical_columns must be provided"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        sample: pd.Series = self.df.iloc[idx]

        masked_sample: pd.Series = sample.copy()
        # Randomly select ONLY CATEGORICAL columns to mask (per paper)
        # Paper: "MLM randomly selects k% features from index 1 to m and masks them as missing"
        categorical_mask = (
            np.random.random_sample(size=len(self.categorical_columns)) < self.mask_prob
        )

        # Apply masking: set masked positions to -1
        if self.categorical_columns:
            for i, col in enumerate(self.categorical_columns):
                if categorical_mask[i]:
                    masked_sample[col] = -1  # Use -1 as mask token

        # Numerical columns are NOT masked in MLM pretraining
        return {
            "masked_categorical": torch.tensor(
                masked_sample[list(self.categorical_columns)].to_numpy(),
                dtype=torch.long,
            )
            if self.categorical_columns
            else torch.empty(0),
            "masked_continuous": torch.tensor(
                masked_sample[list(self.numerical_columns)].to_numpy(),
                dtype=torch.float,
            )
            if self.numerical_columns
            else torch.empty(0),
            "original_categorical": torch.tensor(
                sample[list(self.categorical_columns)].to_numpy(),
                dtype=torch.long,
            )
            if self.categorical_columns
            else torch.empty(0),
            "original_continuous": torch.tensor(
                sample[list(self.numerical_columns)].to_numpy(),
                dtype=torch.float,
            )
            if self.numerical_columns
            else torch.empty(0),
            "mask": torch.tensor(categorical_mask, dtype=torch.bool)
            if self.categorical_columns
            else torch.empty(0, dtype=torch.bool),
        }


class TabularDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        categorical_columns: list[str] | None = None,
        numerical_columns: Optional[list[str]] = None,
        compute_attorney_specialization=False,
        mask_prob: float = 0.15,
        numerical_mask_type: Literal["random", "null_token", "mean"] = "null_token",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        super().save_hyperparameters()

        if data_dir is not None:
            data_dir = Path(data_dir)
            self.train_df = pd.read_csv(data_dir / "train.csv")
            self.val_df = pd.read_csv(data_dir / "valid.csv")
            self.test_df = pd.read_csv(data_dir / "test.csv")
        else:
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
        self.categorical_columns = categorical_columns or []
        self.numerical_columns = numerical_columns if numerical_columns else []
        self.mask_prob = mask_prob
        self.numerical_mask_type = cast(
            Literal["random", "null_token", "mean"], numerical_mask_type
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.continuous_mean_std = None

        self.train_dataset: Optional[MaskedTabularDataset] = None
        self.val_dataset: Optional[MaskedTabularDataset] = None
        self.test_dataset: Optional[MaskedTabularDataset] = None
        self.compute_attorney_specialization = compute_attorney_specialization
        # if self.compute_attorney_specialization:
        #     self.categorical_columns.remove("CaseAttorneyJuris")
        self.categorical_encoder = CategoricalEncoder(self.categorical_columns)

        if self.numerical_columns:
            self.train_df, numerical_means = self._coerce_numerical(self.train_df)
            self.val_df, _ = self._coerce_numerical(self.val_df, numerical_means)
            self.test_df, _ = self._coerce_numerical(self.test_df, numerical_means)

        if self.numerical_columns and self.numerical_mask_type == "mean":
            # Compute mean and std. dev for each numerical column
            if self.train_df is not None:
                mean_std_df = pd.concat(
                    [
                        self.train_df[self.numerical_columns].mean(),
                        self.train_df[self.numerical_columns].std(),
                    ],
                    axis=1,
                )
                self.continuous_mean_std = mean_std_df.to_dict()

    def _coerce_numerical(
        self, df: pd.DataFrame, means: pd.Series | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        for col in self.numerical_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[self.numerical_columns] = df[self.numerical_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        if means is None:
            means = df[self.numerical_columns].mean()
        fill_values = means.fillna(0.0)
        df[self.numerical_columns] = df[self.numerical_columns].fillna(fill_values)
        return df, means

    def setup(
        self, stage: Optional[Literal["fit", "validate", "test", "predict"]] = None
    ) -> None:
        # Only fit encoder once
        if self.categorical_columns and not self.categorical_encoder.encoders:
            self.categorical_encoder = CategoricalEncoder(self.categorical_columns)
            self.categorical_encoder.fit(self.train_df)

        if stage == "fit" or stage is None:
            if self.train_dataset is None:  # Only create once
                self.train_dataset = MaskedTabularDataset(
                self.train_df,
                self.categorical_columns,
                self.numerical_columns,
                self.categorical_encoder,
                self.continuous_mean_std,
                self.mask_prob,
                cast(Literal["random", "null_token", "mean"], self.numerical_mask_type),
                # self.compute_attorney_specialization,
            )
        if stage == "fit" or stage == "validate" or stage is None:
            if self.val_dataset is None:  # Only create once
                self.val_dataset = MaskedTabularDataset(
                self.val_df,
                self.categorical_columns,
                self.numerical_columns,
                self.categorical_encoder,
                self.continuous_mean_std,
                self.mask_prob,
                cast(Literal["random", "null_token", "mean"], self.numerical_mask_type),
                # self.compute_attorney_specialization,
            )
        if stage == "test" or stage is None:
            self.test_dataset = MaskedTabularDataset(
                self.test_df,
                self.categorical_columns,
                self.numerical_columns,
                self.categorical_encoder,
                self.continuous_mean_std,
                self.mask_prob,
                cast(Literal["random", "null_token", "mean"], self.numerical_mask_type),
                # compute_attorney_specialization=self.compute_attorney_specialization,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @cached_property
    def metadata(self):
        return TabularMetaData(self.categorical_encoder, self.numerical_columns)
