"""
YOLO Weights Performance Evaluation
===================================
This script evaluates the performance of all YOLO weights used in the 
hierarchical detection pipeline and generates comprehensive metrics.

Author: Debraj Mukherjee
Project: MIG Weld Defect Detection
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for config import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import weights, confidence_thresholds

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set3")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, '../results'))
TEST_IMAGES_DIR = os.path.abspath(os.path.join(BASE_DIR, '../test_images'))

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_model_info(model_path):
    """Get model information including size and parameters"""
    try:
        model = YOLO(model_path)
        
        # Get model size
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Get model parameters (approximate)
        total_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        return {
            'size_mb': model_size_mb,
            'parameters': total_params,
            'loaded': True
        }
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        return {
            'size_mb': 0,
            'parameters': 0,
            'loaded': False
        }

def benchmark_inference_speed(model_path, test_image_path, num_runs=10):
    """Benchmark inference speed for a model"""
    try:
        model = YOLO(model_path)
        
        # Load test image
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
        else:
            # Create dummy image if test image doesn't exist
            image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Warm-up runs
        for _ in range(3):
            model(image, verbose=False)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            results = model(image, verbose=False)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        return {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times)
        }
    except Exception as e:
        print(f"❌ Error benchmarking {model_path}: {e}")
        return {
            'mean_time_ms': 0,
            'std_time_ms': 0,
            'min_time_ms': 0,
            'max_time_ms': 0
        }

def evaluate_detection_quality(model_path, test_images_dir, confidence_threshold=0.5):
    """Evaluate detection quality on test images"""
    try:
        model = YOLO(model_path)
        
        # Get test images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = []
        
        if os.path.exists(test_images_dir):
            for ext in image_extensions:
                test_images.extend(Path(test_images_dir).glob(f'**/*{ext}'))
        
        if not test_images:
            print(f"⚠️ No test images found in {test_images_dir}")
            return {
                'total_images': 0,
                'total_detections': 0,
                'avg_confidence': 0,
                'avg_detections_per_image': 0
            }
        
        # Limit to first 20 images for performance
        test_images = test_images[:20]
        
        total_detections = 0
        confidence_scores = []
        
        for img_path in test_images:
            try:
                results = model(str(img_path), verbose=False)
                
                if results[0].boxes is not None:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    # Filter by confidence threshold
                    valid_detections = confidences >= confidence_threshold
                    total_detections += np.sum(valid_detections)
                    confidence_scores.extend(confidences[valid_detections])
                    
            except Exception as e:
                print(f"⚠️ Error processing image {img_path}: {e}")
                continue
        
        return {
            'total_images': len(test_images),
            'total_detections': total_detections,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'avg_detections_per_image': total_detections / len(test_images) if test_images else 0
        }
        
    except Exception as e:
        print(f"❌ Error evaluating detection quality for {model_path}: {e}")
        return {
            'total_images': 0,
            'total_detections': 0,
            'avg_confidence': 0,
            'avg_detections_per_image': 0
        }

def create_performance_summary_table(performance_data):
    """Create and save performance summary table"""
    df = pd.DataFrame(performance_data).T
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Model sizes
    model_names = list(performance_data.keys())
    sizes = [performance_data[model]['model_info']['size_mb'] for model in model_names]
    
    axes[0, 0].bar(range(len(model_names)), sizes, color='skyblue')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Size (MB)')
    axes[0, 0].set_title('Model Sizes Comparison', fontweight='bold')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=45)
    
    # 2. Inference times
    inference_times = [performance_data[model]['speed']['mean_time_ms'] for model in model_names]
    
    axes[0, 1].bar(range(len(model_names)), inference_times, color='lightcoral')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Inference Time (ms)')
    axes[0, 1].set_title('Inference Speed Comparison', fontweight='bold')
    axes[0, 1].set_xticks(range(len(model_names)))
    axes[0, 1].set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=45)
    
    # 3. Average confidence scores
    avg_confidences = [performance_data[model]['detection']['avg_confidence'] for model in model_names]
    
    axes[1, 0].bar(range(len(model_names)), avg_confidences, color='lightgreen')
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Average Confidence')
    axes[1, 0].set_title('Average Detection Confidence', fontweight='bold')
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=45)
    
    # 4. Detections per image
    detections_per_image = [performance_data[model]['detection']['avg_detections_per_image'] for model in model_names]
    
    axes[1, 1].bar(range(len(model_names)), detections_per_image, color='gold')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Detections per Image')
    axes[1, 1].set_title('Average Detections per Image', fontweight='bold')
    axes[1, 1].set_xticks(range(len(model_names)))
    axes[1, 1].set_xticklabels([name.replace('_', '\n') for name in model_names], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'yolo_weights_performance_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed table as CSV
    df.to_csv(os.path.join(RESULTS_DIR, 'yolo_weights_detailed_metrics.csv'))
    
    return df

def generate_weights_report(performance_data, save_path):
    """Generate comprehensive report for YOLO weights"""
    
    report_lines = [
        "="*80,
        "🎯 YOLO WEIGHTS PERFORMANCE EVALUATION REPORT",
        "="*80,
        f"📊 Evaluation Overview:",
        f"   • Total Models Evaluated: {len(performance_data)}",
        f"   • Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "🏗️ Hierarchical Detection Pipeline:",
        "   Stage 1: Weld Plate Detection → Stage 2: Weld Seam Detection",
        "   Stage 3: Multi-Class Defect Detection (5 classes)",
        "",
        "📈 Individual Model Performance:",
    ]
    
    for model_name, data in performance_data.items():
        confidence_threshold = confidence_thresholds.get(model_name, 0.5)
        
        report_lines.extend([
            f"",
            f"🔸 {model_name.upper()}:",
            f"   📁 Model Size: {data['model_info']['size_mb']:.2f} MB",
            f"   ⚡ Inference Speed: {data['speed']['mean_time_ms']:.2f} ± {data['speed']['std_time_ms']:.2f} ms",
            f"   🎯 Confidence Threshold: {confidence_threshold}",
            f"   🔍 Avg Detection Confidence: {data['detection']['avg_confidence']:.3f}",
            f"   📊 Detections per Image: {data['detection']['avg_detections_per_image']:.2f}",
            f"   📷 Images Processed: {data['detection']['total_images']}",
        ])
    
    # Performance rankings
    models_by_speed = sorted(performance_data.items(), 
                           key=lambda x: x[1]['speed']['mean_time_ms'])
    models_by_confidence = sorted(performance_data.items(), 
                                key=lambda x: x[1]['detection']['avg_confidence'], reverse=True)
    
    report_lines.extend([
        "",
        "🏆 Performance Rankings:",
        "",
        "⚡ Fastest Models (Inference Speed):",
    ])
    
    for i, (model_name, data) in enumerate(models_by_speed[:3]):
        report_lines.append(f"   {i+1}. {model_name}: {data['speed']['mean_time_ms']:.2f} ms")
    
    report_lines.extend([
        "",
        "🎯 Highest Confidence Models:",
    ])
    
    for i, (model_name, data) in enumerate(models_by_confidence[:3]):
        report_lines.append(f"   {i+1}. {model_name}: {data['detection']['avg_confidence']:.3f}")
    
    report_lines.extend([
        "",
        "💡 Optimization Recommendations:",
        "   • Use fastest models for real-time applications",
        "   • Adjust confidence thresholds based on precision/recall requirements",
        "   • Consider model ensemble for critical applications",
        "   • Monitor performance degradation over time",
        "",
        "⚠️ Important Notes:",
        "   • Inference times measured on current hardware configuration",
        "   • Detection quality depends on test image representativeness",
        "   • Consider GPU/CPU optimization for deployment",
        "",
        "="*80,
    ])
    
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))

def main():
    """Main execution function"""
    print("🚀 Starting YOLO Weights Performance Evaluation...")
    print("="*60)
    
    # Find a test image for benchmarking
    test_image_path = None
    if os.path.exists(TEST_IMAGES_DIR):
        for root, dirs, files in os.walk(TEST_IMAGES_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image_path = os.path.join(root, file)
                    break
            if test_image_path:
                break
    
    performance_data = {}
    
    print(f"🔍 Evaluating {len(weights)} YOLO models...")
    
    for model_name, model_path in weights.items():
        print(f"\n📊 Evaluating: {model_name}")
        
        if not os.path.exists(model_path):
            print(f"   ❌ Model file not found: {model_path}")
            continue
        
        # Get model information
        print("   📁 Getting model info...")
        model_info = get_model_info(model_path)
        
        if not model_info['loaded']:
            continue
        
        # Benchmark inference speed
        print("   ⚡ Benchmarking inference speed...")
        speed_metrics = benchmark_inference_speed(model_path, test_image_path)
        
        # Evaluate detection quality
        print("   🎯 Evaluating detection quality...")
        detection_metrics = evaluate_detection_quality(
            model_path, TEST_IMAGES_DIR, 
            confidence_thresholds.get(model_name, 0.5)
        )
        
        performance_data[model_name] = {
            'model_info': model_info,
            'speed': speed_metrics,
            'detection': detection_metrics
        }
        
        print(f"   ✅ Completed: {speed_metrics['mean_time_ms']:.2f}ms avg inference")
    
    if not performance_data:
        print("❌ No models could be evaluated. Check model paths and files.")
        return
    
    print("\n📊 Generating performance visualizations...")
    
    # Create performance summary
    performance_df = create_performance_summary_table(performance_data)
    print("   ✅ Performance summary charts")
    
    # Generate comprehensive report
    generate_weights_report(
        performance_data,
        os.path.join(RESULTS_DIR, 'yolo_weights_performance_report.txt')
    )
    print("   ✅ Comprehensive performance report")
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 YOLO WEIGHTS EVALUATION COMPLETE!")
    print("="*60)
    print(f"📁 Results saved in: {RESULTS_DIR}")
    print("\n📊 Quick Summary:")
    
    # Show top performers
    fastest_model = min(performance_data.items(), 
                       key=lambda x: x[1]['speed']['mean_time_ms'])
    most_confident = max(performance_data.items(), 
                        key=lambda x: x[1]['detection']['avg_confidence'])
    
    print(f"   ⚡ Fastest Model: {fastest_model[0]} ({fastest_model[1]['speed']['mean_time_ms']:.2f}ms)")
    print(f"   🎯 Highest Confidence: {most_confident[0]} ({most_confident[1]['detection']['avg_confidence']:.3f})")
    
    print("\n📁 Generated Files:")
    print("   • yolo_weights_performance_summary.png")
    print("   • yolo_weights_detailed_metrics.csv") 
    print("   • yolo_weights_performance_report.txt")

if __name__ == "__main__":
    main()
