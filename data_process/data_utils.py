import torch
from torch_geometric.data import Data, Batch
import numpy as np

def preprocess_prediction_data(samples, graph_data_dir):
    # Load graph metadata 
    stations = load_station_metadata(f"{graph_data_dir}/stations.csv")
    routes = load_route_connections(f"{graph_data_dir}/routes.csv")
    
    processed_samples = []
    for sample in samples:
        # Build graph data 
        site_graph = build_site_graph(sample['route'], stations)
        route_graph = build_route_graph(sample['route'], routes)
        flight_graph = build_flight_graph(sample['route'])
        
        # Normalize time series data 
        ts_a = normalize_series(sample['historical_prices']['series_a'])
        ts_b = normalize_series(sample['historical_prices']['series_b'])
        
        processed_samples.append({
            'site_graph': site_graph,
            'route_graph': route_graph,
            'flight_graph': flight_graph,
            'ts_a': ts_a,
            'ts_b': ts_b
        })
    
    # Batch process graph data 
    site_batch = Batch.from_data_list([s['site_graph'] for s in processed_samples])
    route_batch = Batch.from_data_list([s['route_graph'] for s in processed_samples])
    flight_batch = Batch.from_data_list([s['flight_graph'] for s in processed_samples])
    
    return {
        'site_graph': site_batch,
        'route_graph': route_batch,
        'flight_graph': flight_batch,
        'ts_a': np.array([s['ts_a'] for s in processed_samples]),
        'ts_b': np.array([s['ts_b'] for s in processed_samples]),
        'price_scaler': {'min': 300, 'max': 800}  # Should be obtained from actual training data 
    }

def normalize_series(series):
    # Actual normalization should use statistics from training data 
    return (np.array(series) - 300) / 500
