import argparse
import torch
from torch.utils.data import DataLoader
from model import AirfarePredictor, TrainingPipeline
from data_utils import create_dataloader, load_datasets
from config import DEFAULT_HYPERPARAMS

def parse_args():
    parser = argparse.ArgumentParser(description="Airfare Prediction Training")
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Path to processed dataset directory')
    parser.add_argument('--gnn_dim', type=int, default=128,
                       help='Hidden dimension for GNN layers')
    parser.add_argument('--pred_steps', type=int, default=7,
                       help='Number of days to predict')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs for training')
    parser.add_argument('--use_amp', action='store_true',
                       help='Enable mixed precision training')
    parser.add_argument('--window_size', type=int, default=30,
                       help='Historical lookback window size')
    parser.add_argument('--pyramid_levels', nargs='+', type=int, 
                       default=[7, 14, 21],
                       help='Periodic pyramid levels')
    parser.add_argument('--diffusion_steps', type=int, default=1000,
                       help='Diffusion process steps')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Device configuration 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset 
    train_data, val_data = load_datasets(args.data_dir)
    
    # Create data loaders 
    train_loader = create_dataloader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize model 
    model = AirfarePredictor(
        gnn_dim=args.gnn_dim,
        pred_steps=args.pred_steps,
        pyramid_levels=args.pyramid_levels,
        diffusion_steps=args.diffusion_steps
    )
    
    # Multi-GPU training 
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
    
    # Training pipeline 
    pipeline = TrainingPipeline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        init_lr=args.lr,
        early_stop=DEFAULT_HYPERPARAMS['early_stop']
    )
    
    # Mixed precision training 
    if args.use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        pipeline.scaler = scaler
    
    # Start training 
    pipeline.run(epochs=args.max_epochs)

if __name__ == '__main__':
    main()
