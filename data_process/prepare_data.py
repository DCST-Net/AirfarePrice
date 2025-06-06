"""prepare_data.py - Airfare Prediction Data Preprocessing Script
This script converts raw aviation data into a training-ready format, including:

1. Graph structure data construction (stations/routes/flights)

2. Time-series data preprocessing

3. Dataset splitting and standardization"""
import os
import argparse
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data, InMemoryDataset
from typing import Tuple, Dict
# Raw data column name definitions
STATION_COLS = ['station_id', 'latitude', 'longitude', 'city', 'country']
ROUTE_COLS = ['route_id', 'origin_id', 'destination_id', 'distance', 'avg_duration']
FLIGHT_COLS = ['flight_no', 'departure_date', 'aircraft_type', 'base_price']
PRICE_COLS = ['flight_no', 'query_date', 'price']

def load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data files"""
    stations = pd.read_csv(os.path.join(data_dir, 'graph/stations.csv'), usecols=STATION_COLS)
    routes = pd.read_csv(os.path.join(data_dir, 'graph/routes.csv'), usecols=ROUTE_COLS)
    flights = pd.read_csv(os.path.join(data_dir, 'graph/flights.csv'), usecols=FLIGHT_COLS)
    prices = pd.read_csv(os.path.join(data_dir, 'temporal/prices.csv'), usecols=PRICE_COLS)
    return stations, routes, flights, prices

def build_graph_data(stations: pd.DataFrame,
                     routes: pd.DataFrame,
                    flights: pd.DataFrame) -> Dict[str, Data]:
    """Construct hierarchical graph structure data"""
    
    # Station graph: node features + geographical connections
    station_features = stations[['latitude', 'longitude']].values
    station_edges = _create_geo_edges(stations)  # Create edges based on geographical location
    
    station_graph = Data(
        x=torch.FloatTensor(station_features),
        edge_index=torch.LongTensor(station_edges).t().contiguous()
    )
    
    # Route graph: route connection relationships
    route_edges = routes[['origin_id', 'destination_id']].values
    route_features = _extract_route_features(routes)
    
    route_graph = Data(
        x=torch.FloatTensor(route_features),
        edge_index=torch.LongTensor(route_edges).t().contiguous()
    )
    
    # Flight graph: flight-route associations
    flight_edges, flight_features = _build_flight_graph(flights, routes)
    
    flight_graph = Data(
        x=torch.FloatTensor(flight_features),
        edge_index=torch.LongTensor(flight_edges).t().contiguous()
    )
    
    return {
        'station': station_graph,
        'route': route_graph,
        'flight': flight_graph
    }

def process_temporal_data(prices: pd.DataFrame,
                          window_size: int = 30) -> Dict[str, np.ndarray]:
    """Process time-series data and generate samples"""
    
    # Data normalization
    scaler = MinMaxScaler()
    prices['price_norm'] = scaler.fit_transform(prices[['price']])
    
    # Group by flight number
    grouped = prices.groupby('flight_no')
    
    samples = []
    for flight_no, group in grouped:
        # Generate time series A (fixed departure date)
        series_a = _generate_series_a(group, window_size)
        
        # Generate time series B (fixed purchase interval)
        series_b = _generate_series_b(group, window_size)
        
        # Align samples
        for a, b in zip(series_a, series_b):
            samples.append({
                'flight_no': flight_no,
                'series_a': a,
                'series_b': b,
                'target': a[-1]  # Assume predicting the last day
            })
    
    # Convert to numpy arrays
    return {
        'series_a': np.array([s['series_a'] for s in samples]),
        'series_b': np.array([s['series_b'] for s in samples]),
        'targets': np.array([s['target'] for s in samples]),
        'scaler': {'min': scaler.min_[0], 'max': scaler.scale_[0]}
    }

def save_processed_data(data: dict, output_dir: str):
    """Save processed data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save graph data
    torch.save(data['graphs']['station'], os.path.join(output_dir, 'station_graph.pt'))
    (data['graphs']['route'], os.path.join(output_dir, 'route_graph.pt')) 
    (data['graphs']['flight'], os.path.join(output_dir, 'flight_graph.pt'))
    
    # Save time-series data
    np.savez(os.path.join(output_dir, 'temporal.npz'),
            series_a=data['temporal']['series_a'],
            series_b=data['temporal']['series_b'],
            targets=data['temporal']['targets'])
    
    # Save normalization parameters
    with open(os.path.join(output_dir, 'scaler.json'), 'w') as f:
        json.dump(data['temporal']['scaler'], f)

def _create_geo_edges(stations: pd.DataFrame,
                      threshold: float = 1.0) -> np.ndarray:
    """Create station connection edges based on geographical location (example logic)"""
    # In practice, should be calculated based on real distances
    edges = []
    for i in range(len(stations)):
        for j in range(i+1, len(stations)):
            if np.random.rand() < threshold:  # Simplified example
                edges.append([i, j])
    return np.array(edges)

def _extract_route_features(routes: pd.DataFrame) -> np.ndarray:
    """Extract route features"""
    return routes[['distance', 'avg_duration']].values

def _build_flight_graph(flights: pd.DataFrame,
                       routes: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build flight-route association graph"""
    # Example logic: connect flights with routes
    merged = pd.merge(flights, routes, on='route_id')
    flight_edges = merged[['flight_no', 'route_id']].drop_duplicates().values
    flight_features = merged[['base_price', 'aircraft_type']].values
    return flight_edges, flight_features

def _generate_series_a(group: pd.DataFrame,
                       window_size: int) -> list:
    """Generate time series with fixed departure date"""
    group = group.sort_values('query_date')
    return [group['price_norm'].iloc[i:i+window_size].values
            for i in range(len(group)-window_size)]

def _generate_series_b(group: pd.DataFrame,
                      window_size: int) -> list:
    """Generate time series with fixed purchase interval"""
     # Assuming weekly periodicity
    return [group['price_norm'].iloc[i::7][:window_size].values  # Sample on the same day each week
           for i in range(7)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Raw data directory')
    parser.add_argument('--output_dir', default='processed', help='Output directory')
    parser.add_argument('--window_size', type=int, default=30, help='Time series window size')
    args = parser.parse_args()
    
    # Execute preprocessing pipeline
    stations, routes, flights, prices = load_raw_data(args.data_dir)
    
    graph_data = build_graph_data(stations, routes, flights)
    temporal_data = process_temporal_data(prices, args.window_size)
    
    processed_data = {
        'graphs': graph_data,
        'temporal': temporal_data
    }
    
    save_processed_data(processed_data, args.output_dir)
    print(f"Data preprocessing completed! Results saved to {args.output_dir}")
