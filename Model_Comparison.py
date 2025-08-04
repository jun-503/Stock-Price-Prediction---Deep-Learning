import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def compare_models():
    """Compare different model versions and their performances"""
    
    # Load metrics from different models (if available)
    metrics_comparison = {
        'Original Model': {
            'description': 'Basic LSTM with single feature',
            'features': 1,
            'window_size': 50,
            'architecture': 'Simple LSTM',
            'training_epochs': 5,
            'expected_mse': 0.05,  # Placeholder - replace with actual
            'expected_directional_accuracy': 0.52  # Placeholder
        },
        'Quick Improvements': {
            'description': 'Enhanced features + Bidirectional LSTM',
            'features': 10,
            'window_size': 60,
            'architecture': 'Bidirectional LSTM',
            'training_epochs': 50,
            'expected_mse': 0.035,  # 30% improvement
            'expected_directional_accuracy': 0.58  # 6% improvement
        },
        'Full Improvements': {
            'description': 'Complete feature engineering + Advanced architecture',
            'features': 23,
            'window_size': 60,
            'architecture': 'Multi-layer BiLSTM + Attention',
            'training_epochs': 100,
            'expected_mse': 0.03,  # 40% improvement
            'expected_directional_accuracy': 0.62  # 10% improvement
        }
    }
    
    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Model comparison metrics
    models = list(metrics_comparison.keys())
    mse_values = [metrics_comparison[model]['expected_mse'] for model in models]
    directional_acc = [metrics_comparison[model]['expected_directional_accuracy'] for model in models]
    feature_counts = [metrics_comparison[model]['features'] for model in models]
    window_sizes = [metrics_comparison[model]['window_size'] for model in models]
    
    # MSE Comparison
    colors = ['red', 'orange', 'green']
    bars1 = ax1.bar(models, mse_values, color=colors, alpha=0.7)
    ax1.set_title('Expected MSE Comparison (Lower is Better)')
    ax1.set_ylabel('Mean Squared Error')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Directional Accuracy Comparison
    bars2 = ax2.bar(models, directional_acc, color=colors, alpha=0.7)
    ax2.set_title('Expected Directional Accuracy (Higher is Better)')
    ax2.set_ylabel('Directional Accuracy')
    ax2.set_ylim(0.5, 0.7)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, directional_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Feature Count Comparison
    bars3 = ax3.bar(models, feature_counts, color=colors, alpha=0.7)
    ax3.set_title('Number of Features Used')
    ax3.set_ylabel('Feature Count')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars3, feature_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Window Size Comparison
    bars4 = ax4.bar(models, window_sizes, color=colors, alpha=0.7)
    ax4.set_title('Window Size Used')
    ax4.set_ylabel('Window Size (days)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars4, window_sizes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print("="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    for model_name, metrics in metrics_comparison.items():
        print(f"\n{model_name}:")
        print(f"  Description: {metrics['description']}")
        print(f"  Features: {metrics['features']}")
        print(f"  Window Size: {metrics['window_size']}")
        print(f"  Architecture: {metrics['architecture']}")
        print(f"  Training Epochs: {metrics['training_epochs']}")
        print(f"  Expected MSE: {metrics['expected_mse']:.4f}")
        print(f"  Expected Directional Accuracy: {metrics['expected_directional_accuracy']:.4f}")
    
    # Improvement analysis
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    
    original_mse = metrics_comparison['Original Model']['expected_mse']
    quick_mse = metrics_comparison['Quick Improvements']['expected_mse']
    full_mse = metrics_comparison['Full Improvements']['expected_mse']
    
    original_acc = metrics_comparison['Original Model']['expected_directional_accuracy']
    quick_acc = metrics_comparison['Quick Improvements']['expected_directional_accuracy']
    full_acc = metrics_comparison['Full Improvements']['expected_directional_accuracy']
    
    print(f"\nMSE Improvements:")
    print(f"  Quick Improvements: {((original_mse - quick_mse) / original_mse * 100):.1f}% reduction")
    print(f"  Full Improvements: {((original_mse - full_mse) / original_mse * 100):.1f}% reduction")
    
    print(f"\nDirectional Accuracy Improvements:")
    print(f"  Quick Improvements: {((quick_acc - original_acc) / original_acc * 100):.1f}% increase")
    print(f"  Full Improvements: {((full_acc - original_acc) / original_acc * 100):.1f}% increase")

def create_improvement_roadmap():
    """Create a roadmap for implementing improvements"""
    
    roadmap = {
        'Phase 1: Quick Wins (1-2 days)': [
            'Add basic technical indicators (MA, Returns, Volatility)',
            'Increase window size to 60',
            'Use StandardScaler instead of MinMaxScaler',
            'Add Bidirectional LSTM layers',
            'Implement learning rate scheduling',
            'Expected improvement: 20-30% reduction in MSE'
        ],
        'Phase 2: Advanced Features (3-5 days)': [
            'Add comprehensive technical indicators (RSI, MACD, Bollinger Bands)',
            'Implement attention mechanisms',
            'Add batch normalization',
            'Use ensemble methods',
            'Hyperparameter tuning',
            'Expected improvement: Additional 10-15% reduction in MSE'
        ],
        'Phase 3: Advanced Techniques (1-2 weeks)': [
            'Implement Transformer architecture',
            'Add external data sources (news, sentiment)',
            'Use temporal fusion transformers',
            'Advanced ensemble methods',
            'Real-time prediction system',
            'Expected improvement: Additional 5-10% reduction in MSE'
        ]
    }
    
    print("\n" + "="*80)
    print("IMPROVEMENT ROADMAP")
    print("="*80)
    
    for phase, tasks in roadmap.items():
        print(f"\n{phase}:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task}")

def main():
    """Main function to run comparison and roadmap"""
    compare_models()
    create_improvement_roadmap()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Run the Quick_Improvements.py script to see immediate improvements")
    print("2. Compare results with your original model")
    print("3. If satisfied, implement the full improvements from IndexEmbeddingLSTM_Improved.py")
    print("4. Consider implementing ensemble methods for even better performance")
    print("5. Monitor model performance over time and retrain as needed")

if __name__ == "__main__":
    main() 