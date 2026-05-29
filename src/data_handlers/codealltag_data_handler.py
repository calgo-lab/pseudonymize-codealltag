from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict
from pandas import DataFrame
from sklearn.model_selection import KFold

import json

import numpy as np
import pandas as pd


class CodealltagDataHandler:
    """
    Data handler for CodE Alltag dataset.
    """
    def __init__(self,
                 project_root: Path,
                 data_dir: Path = None) -> None:
            
        """
        Initializes the data handler for CodE Alltag dataset.

        :param project_root: The root directory of the project.
        :param data_dir: Optional path to the data directory. If not provided, it defaults to 'data' directory under the project root.

        :return: None
        """

        self.project_root = project_root

        if data_dir:
            self.data_dir: Path = data_dir
        else:
            self.data_dir: Path = self.project_root / "data"
        
        self._email_files_info_dataframe_file_path: Path = self.data_dir / "email_files_info_10k_with_text_and_annotations.parquet"
        
    def get_email_files_info_10k_dataframe(self) -> DataFrame:
        """
        Get the email files info dataframe
        
        :return: DataFrame with email files info.
        """
        
        if (not self._email_files_info_dataframe_file_path.exists()):
            raise FileNotFoundError(f"File not found at {self._email_files_info_dataframe_file_path}.")
        
        return pd.read_parquet(self._email_files_info_dataframe_file_path, engine="pyarrow")
    

    def get_train_dev_test_datasetdict(self, 
                                       random_state: int = 2026, 
                                       k: int = 1) -> DatasetDict:
        
        """
        Retrieve the train, dev, and test dataframes for the specified fold.
        :param random_state: Random state for reproducibility.
        :param k: The fold number to retrieve (1-based index).

        :return: A DatasetDict containing the train, dev, and test datasets.
        """

        sample_df = self.get_email_files_info_10k_dataframe()
        sample_df.insert(0, "ID", sample_df.index.astype(int))

        sample_df.reset_index(drop=True, inplace=True)

        fold_tuples = list()
        splits = list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(sample_df.index.to_numpy()))
        train_dev_test_k_folds = self.get_train_dev_test_folds()
        for index, fold in enumerate(train_dev_test_k_folds):
            train_indices = list()
            fold_train_indices = fold[1]
            for fold_train_index in fold_train_indices:
                train_indices += list(splits[fold_train_index][1])
            dev_indices = list()
            fold_dev_indices = fold[2]
            for fold_dev_index in fold_dev_indices:
                dev_indices += list(splits[fold_dev_index][1])
            test_indices = list()
            fold_test_indices = fold[3]
            for fold_test_index in fold_test_indices:
                test_indices += list(splits[fold_test_index][1])
            fold_tuples.append((
                index + 1,
                sample_df[sample_df.index.isin(train_indices)].ID.tolist(),
                sample_df[sample_df.index.isin(dev_indices)].ID.tolist(),
                sample_df[sample_df.index.isin(test_indices)].ID.tolist()
            ))
        
        kth_tuple = fold_tuples[k-1]

        train_df = sample_df[sample_df.ID.isin(kth_tuple[1])]
        train_df = train_df.sort_values(by="ID").reset_index(drop=True)
        train_ds = Dataset.from_pandas(train_df)

        dev_df = sample_df[sample_df.ID.isin(kth_tuple[2])]
        dev_df = dev_df.sort_values(by="ID").reset_index(drop=True)
        dev_ds = Dataset.from_pandas(dev_df)

        test_df = sample_df[sample_df.ID.isin(kth_tuple[3])]
        test_df = test_df.sort_values(by="ID").reset_index(drop=True)
        test_ds = Dataset.from_pandas(test_df)

        return DatasetDict({
            "train": train_ds, 
            "dev": dev_ds, 
            "test": test_ds
        })
        
    
    def get_fold_stats(self, 
                       fold_datasetdict: DatasetDict, 
                       label_order: List[str]) -> Dict[str, str]:
        """
        Given a DatasetDict with 'train', 'dev', 'test' splits,
        returns a dict with total files, tokens, entities,
        and per-label entity counts for each split.

        :param fold_datasetdict: The DatasetDict containing 'train', 'dev', 'test' datasets.
        :param label_order: The order of labels to display in the stats.
        
        :return: A dictionary with stats as keys and formatted strings as values
        """
        train_df = fold_datasetdict["train"].to_pandas()
        dev_df = fold_datasetdict["dev"].to_pandas()
        test_df = fold_datasetdict["test"].to_pandas()

        stats: Dict[str, str] = dict()
        stats["total_files"] = {
            "train": len(train_df['ID'].unique()),
            "dev": len(dev_df['ID'].unique()),
            "test": len(test_df['ID'].unique())
        }

        stats["total_tokens"] = {
            "train": sum(train_df['token_count']),
            "dev": sum(dev_df['token_count']),
            "test": sum(test_df['token_count'])
        }
        stats["total_entities"] = {
            "train": sum(train_df['entity_count']),
            "dev": sum(dev_df['entity_count']),
            "test": sum(test_df['entity_count'])
        }

        stats["train_files"] = train_df['file_path'].unique().tolist()
        stats["dev_files"] = dev_df['file_path'].unique().tolist()
        stats["test_files"] = test_df['file_path'].unique().tolist()

        train_counts = self._aggregate_label_counts(train_df)
        dev_counts = self._aggregate_label_counts(dev_df)
        test_counts = self._aggregate_label_counts(test_df)

        for label in label_order:
            train_val = train_counts.get(label, 0)
            dev_val = dev_counts.get(label, 0)
            test_val = test_counts.get(label, 0)
            stats[label] = {
                "train": train_val, 
                "dev": dev_val, 
                "test": test_val
            }

        return stats
    
    @staticmethod
    def get_train_dev_test_folds(n_fold: int = 5, 
                                 train_percent: float = 0.6, 
                                 dev_percent: float = 0.2) -> List[Tuple]:
        fold_tuples = list()
        indices = list(range(n_fold))
        train_start = 0
        train_end = int(round(n_fold * train_percent))
        dev_start = train_end
        dev_end = int(round(n_fold * (train_percent + dev_percent)))
        test_start = dev_end
        test_end = n_fold
        for index in indices:
            rolled_indices = np.roll(indices, -index)
            train_indices = list(rolled_indices[train_start: train_end])
            dev_indices = list(rolled_indices[dev_start: dev_end])
            test_indices = list(rolled_indices[test_start: test_end])
            fold_tuples.append((
                index + 1,
                train_indices,
                dev_indices,
                test_indices
            ))
        return fold_tuples
    
    @staticmethod
    def _aggregate_label_counts(input_df: DataFrame, 
                                column_name: str = "label_wise_entity_count"):
        total = Counter()
        for item in input_df[column_name]:
            total.update(json.loads(item))
        return total
