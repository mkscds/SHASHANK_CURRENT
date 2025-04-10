
"""Partition the data and create the dataloaders."""

from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from .dataset_preparation import (partition_data, partition_data_dirichlet, partition_data_label_quantity)

# def load_datasets(config: DictConfig, num_clients: int, val_ratio: float = 0.1, seed: Optional[int] = 42) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, DataLoader]:
#     """Create the dataloaders to be fed into the model.
    
#     Parameters
#     ----------

#     config: DictConfig
#         Parameterises the dataset partitioning process
    
#     num_clients: int
#         The number of clients that hold a part of the data

#     val_ratio: float, Optional
#         The ratio of training data that will be used for validation (between 0 and 1)
#         by default 0.1

#     seed: int, Optional
#         Used to set a fix seed to replicate experiments, by default  42
    
#     Returns
#     -------
#     Tuple[List[DataLoader], List[DataLoader], DataLoader, DataLoader]
#     """
#     print(f" Dataset Partitioining config: {config}")
#     partitioning = ""
#     if "partitioning" in config:
#         partitioning = config.partitioning
    
#     if partitioning == "dirichlet":
#         alpha = 0.5
#         if "alpha" in config:
#             alpha = config.alpha

#         datasets, testset, serverset = partition_data_dirichlet(num_clients, alpha=alpha, seed=seed, dataset_name=config.name)
    
#     elif partitioning == "label_quantity":
#         labels_per_client = 2
#         if "labels_per_client" in config:
#             labels_per_client = config.labels_per_client
        
#         datasets, testset, serverset = partition_data_label_quantity(num_clients, labels_per_client=labels_per_client, seed=seed, dataset_name=config.name)

#     elif partitioning == "iid":
#         datasets, testset, serverset = partition_data(num_clients, similarity=1.0, seed=seed, dataset_name=config.name)

#     elif partitioning == "iid_noniid":
#         similarity = 0.5
#         if "similarity" in config:
#             similarity = config.similarity

#         datasets, testset, serverset = partition_data(num_clients, similarity=similarity, seed=seed, dataset_name=config.name)

#     batch_size = -1
#     if "batch_size" in config:
#         batch_size = config.batch_size

#     elif "batch_size_ratio" in config:
#         batch_size_ratio = config.batch_size_ratio

#     else:
#         raise ValueError
    
#     # split each partitioning into train/val and create DataLoader
#     trainloaders = []
#     valloaders = []
#     for dataset in datasets:
#         len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
#         lengths = [len(dataset) - len_val, len_val]
#         ds_train, ds_val = random_split(dataset, lengths, torch.Generator().manual_seed(seed))
        
#         if batch_size == -1:
#             batch_size = int(len(ds_train) * batch_size_ratio)
        
#         trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
#         valloaders.append(DataLoader(ds_val, batch_size=batch_size))

#     return trainloaders, valloaders, DataLoader(testset, batch_size=len(testset)), DataLoader(serverset, batch_size=batch_size, shuffle=True)



def load_datasets(args) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model."""
    print(f"Dataset Partitioning config: {args}")
    
    partitioning = args.partitioning
    
    if partitioning == "dirichlet":
        datasets, testset, serverset = partition_data_dirichlet(args.num_clients, alpha=args.alpha, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "label_quantity":
        datasets, testset, serverset = partition_data_label_quantity(args.num_clients, labels_per_client=args.labels_per_client, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "iid":
        datasets, testset, serverset = partition_data(args.num_clients, similarity=1.0, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "iid_noniid":
        datasets, testset, serverset = partition_data(args.num_clients, similarity=args.similarity, seed=args.seed, dataset_name=args.name)
    
    batch_size = args.batch_size if args.batch_size else int(len(datasets[0]) * args.batch_size_ratio)
    
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) * args.val_ratio) if args.val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(dataset, lengths, torch.Generator().manual_seed(args.seed))
        
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    
    return trainloaders, valloaders, DataLoader(testset, batch_size=len(testset)), DataLoader(serverset, batch_size=batch_size, shuffle=True)


def load_datasets_fedavg(args) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model."""
    print(f"Dataset Partitioning config: {args}")
    
    partitioning = args.partitioning
    
    if partitioning == "dirichlet":
        datasets, testset = partition_data_dirichlet(args.num_clients, alpha=args.alpha, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "label_quantity":
        datasets, testset = partition_data_label_quantity(args.num_clients, labels_per_client=args.labels_per_client, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "iid":
        datasets, testset = partition_data(args.num_clients, similarity=1.0, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "iid_noniid":
        datasets, testset = partition_data(args.num_clients, similarity=args.similarity, seed=args.seed, dataset_name=args.name)
    
    batch_size = args.batch_size if args.batch_size else int(len(datasets[0]) * args.batch_size_ratio)
    
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) * args.val_ratio) if args.val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(dataset, lengths, torch.Generator().manual_seed(args.seed))
        
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    
    return trainloaders, valloaders, DataLoader(testset, batch_size=len(testset))
        
def load_datasets_FEDMD(args) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model."""
    print(f"Dataset Partitioning config: {args}")
    
    partitioning = args.partitioning
    
    if partitioning == "dirichlet":
        datasets, testset = partition_data_dirichlet(args.num_clients, alpha=args.alpha, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "label_quantity":
        datasets, testset = partition_data_label_quantity(args.num_clients, labels_per_client=args.labels_per_client, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "iid":
        datasets, testset = partition_data(args.num_clients, similarity=1.0, seed=args.seed, dataset_name=args.name)
    
    elif partitioning == "iid_noniid":
        datasets, testset = partition_data(args.num_clients, similarity=args.similarity, seed=args.seed, dataset_name=args.name)
    
    batch_size = args.batch_size if args.batch_size else int(len(datasets[0]) * args.batch_size_ratio)
    
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) * args.val_ratio) if args.val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(dataset, lengths, torch.Generator().manual_seed(args.seed))
        
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    
    return trainloaders, valloaders, DataLoader(testset, batch_size=len(testset))