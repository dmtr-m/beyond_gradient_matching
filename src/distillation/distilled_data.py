from dataclasses import dataclass


@dataclass
class DistilledDataConfig:
    dpc: int
    n_dataset: int
    over_sample_ratio: float = 1.0  # if > 1.0, prune samples with k_center
    n_eval_dataset: int = 5  # datasets generated per validation call (5 → 1 speeds up LM training significantly)
    save_dataset_path: str = "path/to/save_dataset_dir"
