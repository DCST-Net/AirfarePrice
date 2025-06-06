def dataloader(dataset: dict, 
               batch_size: int = 32,
               shuffle: bool = True):
    """
    Creates a PyTorch data loader.
    
    Args:
        dataset: A dictionary containing the following keys

            - site_graphs: List of site graph data

            - route_graphs: List of route graph data

            - flight_graphs: List of flight graph data

            - ts_a: Time series A data [num_samples, seq_len_a]

            - ts_b: Time series B data [num_samples, seq_len_b]

            - targets: Target prices [num_samples, pred_steps]
    """
    # Convert to PyG Batch objects
    site_batch = Batch.from_data_list(dataset['site_graphs'])
    route_batch = Batch.from_data_list(dataset['route_graphs'])
    flight_batch = Batch.from_data_list(dataset['flight_graphs'])
    
    # Create TensorDataset
    full_dataset = TensorDataset(
        site_batch,
        route_batch,
        flight_batch,
        torch.FloatTensor(dataset['ts_a']),
        torch.FloatTensor(dataset['ts_b']),
        torch.FloatTensor(dataset['targets'])
    )
    
    return DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
