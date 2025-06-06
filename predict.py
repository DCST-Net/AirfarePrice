import argparse
import json
import torch
import pandas as pd
import numpy as np
from model import AirfarePredictor
from data_utils import preprocess_prediction_data

def parse_args():
    parser = argparse.ArgumentParser(description="Airfare Price Prediction")
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_data', type=str, required=True,
                       help='Path to input JSON file')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output file path for predictions')
    parser.add_argument('--graph_data_dir', type=str, default='data/graph',
                       help='Directory containing graph metadata')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    model = AirfarePredictor()
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Process weights from multi-GPU training
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = load_model(args.model_checkpoint, device)
    
    # Load and preprocess input data
    with open(args.input_data) as f:
        input_data = json.load(f)
    
    # Preprocess prediction data
    processed_data = preprocess_prediction_data(
        input_data['samples'],
        args.graph_data_dir
    )
    
    # Convert to Tensor
    site_data = processed_data['site_graph'].to(device)
    route_data = processed_data['route_graph'].to(device)
    flight_data = processed_data['flight_graph'].to(device)
    ts_a = torch.FloatTensor(processed_data['ts_a']).to(device)
    ts_b = torch.FloatTensor(processed_data['ts_b']).to(device)
    
    # Perform prediction
    with torch.no_grad():
        graph_feat, _ = model.gnn(site_data, route_data, flight_data)
        trend_pred, periodic_pred = model.decoder(ts_a, ts_b, graph_feat)
        final_pred = model.fusion(trend_pred, periodic_pred)
    
    # Convert to actual prices
    min_price = processed_data['price_scaler']['min']
    max_price = processed_data['price_scaler']['max']
    predictions = final_pred.cpu().numpy() * (max_price - min_price) + min_price
    
    # Save results
    results = []
    for sample, pred in zip(input_data['samples'], predictions):
        result = {
            'route': sample['route'],
            'departure_date': sample['departure_date'],
            'predictions': {}
        }
        for i, price in enumerate(pred, 1):
            result['predictions'][f'day_{i}'] = round(price, 2)
        results.append(result)
    
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()
