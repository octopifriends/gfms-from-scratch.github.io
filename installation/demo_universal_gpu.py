#!/usr/bin/env python3
"""
Demo: Universal GPU Usage in GeoAI
Shows how to use GPU acceleration across Mac (MPS) and Linux (CUDA) platforms
"""

import torch
import numpy as np
import time
import platform

def setup_device():
    """Universal device setup for Mac MPS and Linux CUDA"""
    print("🔧 Setting up compute device...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Using CUDA GPU: {device_name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Silicon GPU (MPS)")
        print(f"   System: {platform.platform()}")
        
    else:
        device = torch.device('cpu')
        print("✅ Using CPU")
        cores = torch.get_num_threads()
        print(f"   Threads: {cores}")
    
    return device

def demo_basic_operations(device):
    """Demo basic tensor operations on the selected device"""
    print(f"\n🧪 Testing basic operations on {device}...")
    
    # Create some sample geospatial-like data
    batch_size = 32
    channels = 3  # RGB bands
    height, width = 256, 256  # Typical satellite image patch
    
    print(f"Creating sample data: {batch_size}x{channels}x{height}x{width}")
    
    # Simulate satellite imagery batch
    start_time = time.time()
    
    satellite_images = torch.randn(batch_size, channels, height, width, device=device)
    
    # Simulate some processing (like convolution for feature extraction)
    # This mimics what you'd do in a CNN for land use classification
    kernel = torch.randn(64, channels, 3, 3, device=device)  # 64 filters, 3x3 kernel
    
    # Convolution operation (common in geospatial CNNs)
    if device.type == 'mps':
        # MPS may need explicit padding
        padding = 1
        output = torch.nn.functional.conv2d(satellite_images, kernel, padding=padding)
    else:
        output = torch.nn.functional.conv2d(satellite_images, kernel, padding=1)
    
    # Force computation to complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    end_time = time.time()
    duration = (end_time - start_time) * 1000
    
    print(f"✅ Convolution completed in {duration:.2f} ms")
    print(f"   Input shape: {satellite_images.shape}")
    print(f"   Output shape: {output.shape}")
    
    return output

def demo_geoai_workflow(device):
    """Demo a simplified GeoAI workflow"""
    print(f"\n🌍 GeoAI Workflow Demo on {device}...")
    
    # Simulate loading a pre-trained model (like ResNet for land use classification)
    print("📥 Loading model...")
    
    # Simple CNN for demonstration
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 10)  # 10 land use classes
    ).to(device)
    
    print(f"✅ Model loaded on {device}")
    
    # Simulate a batch of satellite image patches
    batch_size = 16
    satellite_batch = torch.randn(batch_size, 3, 64, 64, device=device)
    
    print(f"🛰️  Processing {batch_size} satellite image patches...")
    
    # Inference
    start_time = time.time()
    
    with torch.no_grad():
        predictions = model(satellite_batch)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
    
    # Force computation to complete
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    
    print(f"✅ Inference completed in {inference_time:.2f} ms")
    print(f"   Throughput: {batch_size / (inference_time / 1000):.1f} images/second")
    
    # Show sample predictions
    predicted_classes = torch.argmax(probabilities, dim=1)
    confidence_scores = torch.max(probabilities, dim=1).values
    
    print(f"\n📊 Sample predictions:")
    land_use_classes = ['Urban', 'Forest', 'Water', 'Agriculture', 'Desert', 
                       'Mountains', 'Beach', 'Snow', 'Clouds', 'Barren']
    
    for i in range(min(3, batch_size)):
        class_idx = predicted_classes[i].item()
        confidence = confidence_scores[i].item()
        print(f"   Image {i+1}: {land_use_classes[class_idx]} ({confidence:.3f})")

def demo_memory_management(device):
    """Demo memory management across different devices"""
    print(f"\n💾 Memory Management Demo on {device}...")
    
    if device.type == 'cuda':
        # CUDA memory management
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e6
        cached = torch.cuda.memory_reserved() / 1e6
        print(f"   CUDA Memory - Allocated: {allocated:.1f} MB, Cached: {cached:.1f} MB")
        
    elif device.type == 'mps':
        # MPS doesn't have detailed memory APIs yet
        print("   MPS memory management is handled automatically")
        
    else:
        # CPU memory
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"   System Memory - Available: {memory_info.available / 1e9:.1f} GB")

def main():
    """Main demo function"""
    print("🌍 Universal GPU Demo for GeoAI")
    print("=" * 40)
    
    # Setup device
    device = setup_device()
    
    # Run demos
    demo_basic_operations(device)
    demo_geoai_workflow(device)
    demo_memory_management(device)
    
    print(f"\n🎉 Demo completed successfully on {device}!")
    print(f"\n💡 Usage in your GeoAI projects:")
    print(f"   1. Use the universal device setup at the start of your scripts")
    print(f"   2. Always move models and data to the same device")
    print(f"   3. Use appropriate synchronization for timing measurements")
    print(f"   4. Handle device-specific optimizations when needed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you have the geoAI environment activated")