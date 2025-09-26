"""
Visualization script for geospatial prediction hierarchy.
Creates diagrams showing the relationships between different prediction tasks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

def create_prediction_hierarchy_diagram():
    """Create a comprehensive diagram showing the hierarchy of prediction tasks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Left panel: Pixel Values vs Labels
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Fundamental Data Types in Geospatial Analysis', fontsize=16, fontweight='bold')
    
    # Pixel Values Box
    pixel_box = FancyBboxPatch((0.5, 6), 4, 3,
                               boxstyle="round,pad=0.1",
                               facecolor='lightblue',
                               edgecolor='darkblue',
                               linewidth=2)
    ax1.add_patch(pixel_box)
    ax1.text(2.5, 7.5, 'Pixel Values', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(2.5, 7, '(Continuous)', ha='center', va='center', fontsize=10, style='italic')
    ax1.text(2.5, 6.5, '• Spectral bands\n• Physical quantities\n• Indices', 
             ha='center', va='center', fontsize=9)
    
    # Labels Box
    label_box = FancyBboxPatch((5.5, 6), 4, 3,
                               boxstyle="round,pad=0.1",
                               facecolor='lightgreen',
                               edgecolor='darkgreen',
                               linewidth=2)
    ax1.add_patch(label_box)
    ax1.text(7.5, 7.5, 'Labels', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(7.5, 7, '(Categorical)', ha='center', va='center', fontsize=10, style='italic')
    ax1.text(7.5, 6.5, '• Classes\n• Objects\n• Boundaries', 
             ha='center', va='center', fontsize=9)
    
    # Temporal arrow
    temporal_arrow = FancyArrowPatch((1, 5.5), (1, 3.5),
                                   connectionstyle="arc3,rad=0",
                                   arrowstyle="->",
                                   mutation_scale=20,
                                   color='darkblue')
    ax1.add_patch(temporal_arrow)
    ax1.text(0.5, 4.5, 'Time →\nNext Value\n(GPT)', ha='center', va='center', 
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
    
    # Spatial arrow
    spatial_arrow = FancyArrowPatch((4, 5.5), (4, 3.5),
                                  connectionstyle="arc3,rad=0",
                                  arrowstyle="->",
                                  mutation_scale=20,
                                  color='darkblue')
    ax1.add_patch(spatial_arrow)
    ax1.text(4, 4.5, 'Space →\nMissing Values\n(BERT)', ha='center', va='center',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white'))
    
    # Label tasks
    tasks = [
        ('Image\nClassification', 6, 4.5),
        ('Pixel-wise\nClassification', 7.5, 4.5),
        ('Object\nDetection', 6, 3),
        ('Object\nSegmentation', 7.5, 3),
        ('Regression\n(Pixel-wise)', 9, 3.75)
    ]
    
    for task, x, y in tasks:
        task_box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor='lightyellow',
                                 edgecolor='orange',
                                 linewidth=1)
        ax1.add_patch(task_box)
        ax1.text(x, y, task, ha='center', va='center', fontsize=8)
    
    # Connect labels to tasks
    for x in [6, 7.5]:
        for y in [4.5, 3]:
            arrow = FancyArrowPatch((7.5, 6), (x, y+0.3),
                                  connectionstyle="arc3,rad=0.2",
                                  arrowstyle="->",
                                  mutation_scale=15,
                                  color='darkgreen',
                                  alpha=0.5)
            ax1.add_patch(arrow)
    
    # Right panel: Task Hierarchy
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Task Granularity Hierarchy', fontsize=16, fontweight='bold')
    
    # Create hierarchy levels
    levels = [
        ('Scene Level', 8, 'Image Classification', 'lightcoral'),
        ('Object Level', 6, 'Detection + Segmentation', 'lightsalmon'),
        ('Pixel Level', 4, 'Classification + Regression', 'lightgoldenrodyellow'),
        ('Sub-pixel Level', 2, 'Super-resolution + Unmixing', 'lightcyan')
    ]
    
    y_prev = 10
    for level_name, y_pos, tasks, color in levels:
        # Level box
        level_box = FancyBboxPatch((1, y_pos-0.5), 8, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=color,
                                  edgecolor='black',
                                  linewidth=2)
        ax2.add_patch(level_box)
        ax2.text(2, y_pos, level_name, ha='left', va='center', fontsize=12, fontweight='bold')
        ax2.text(8, y_pos, tasks, ha='right', va='center', fontsize=10, style='italic')
        
        # Connect levels
        if y_prev < 10:
            arrow = FancyArrowPatch((5, y_prev-0.5), (5, y_pos+0.5),
                                  connectionstyle="arc3,rad=0",
                                  arrowstyle="->",
                                  mutation_scale=20,
                                  color='gray')
            ax2.add_patch(arrow)
        y_prev = y_pos
    
    # Add complexity gradient
    ax2.text(0.5, 9, 'Low\nComplexity', ha='center', va='center', fontsize=10, rotation=90)
    ax2.text(0.5, 3, 'High\nComplexity', ha='center', va='center', fontsize=10, rotation=90)
    
    # Add granularity gradient
    ax2.text(9.5, 9, 'Coarse\nGranularity', ha='center', va='center', fontsize=10, rotation=270)
    ax2.text(9.5, 3, 'Fine\nGranularity', ha='center', va='center', fontsize=10, rotation=270)
    
    plt.tight_layout()
    return fig

def create_input_output_diagram():
    """Create a diagram showing input-output relationships for different tasks."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Define task configurations
    tasks = [
        {
            'title': 'Temporal Prediction',
            'input': 'Time Series Stack\n(T × H × W × B)',
            'model': 'Transformer\n(GPT-style)',
            'output': 'Next Timestep\n(H × W × B)',
            'color': 'lightblue'
        },
        {
            'title': 'Spatial Interpolation',
            'input': 'Masked Image\n(H × W × B)',
            'model': 'Transformer\n(BERT-style)',
            'output': 'Complete Image\n(H × W × B)',
            'color': 'lightgreen'
        },
        {
            'title': 'Image Classification',
            'input': 'Full Scene\n(H × W × B)',
            'model': 'CNN/ViT\nEncoder',
            'output': 'Scene Label\n(1 × C classes)',
            'color': 'lightcoral'
        },
        {
            'title': 'Pixel Classification',
            'input': 'Image Patches\n(H × W × B)',
            'model': 'U-Net/\nSegFormer',
            'output': 'Label Map\n(H × W × C)',
            'color': 'lightyellow'
        },
        {
            'title': 'Object Detection',
            'input': 'Full Image\n(H × W × B)',
            'model': 'YOLO/\nDETR',
            'output': 'Bboxes + Classes\n(N × [x,y,w,h,c])',
            'color': 'lightcyan'
        },
        {
            'title': 'Biophysical Regression',
            'input': 'Multi-band Stack\n(H × W × B)',
            'model': 'CNN/FM\n+ MLP',
            'output': 'Parameter Maps\n(H × W × P)',
            'color': 'lavender'
        }
    ]
    
    for ax, task in zip(axes, tasks):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(task['title'], fontsize=12, fontweight='bold')
        
        # Input box
        input_box = FancyBboxPatch((0.5, 6), 2.5, 3,
                                  boxstyle="round,pad=0.1",
                                  facecolor=task['color'],
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(input_box)
        ax.text(1.75, 7.5, 'Input', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(1.75, 6.8, task['input'], ha='center', va='center', fontsize=8)
        
        # Model box
        model_box = FancyBboxPatch((3.75, 6), 2.5, 3,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightgray',
                                  edgecolor='black',
                                  linewidth=2)
        ax.add_patch(model_box)
        ax.text(5, 7.5, 'Model', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(5, 6.8, task['model'], ha='center', va='center', fontsize=8)
        
        # Output box
        output_box = FancyBboxPatch((7, 6), 2.5, 3,
                                   boxstyle="round,pad=0.1",
                                   facecolor=task['color'],
                                   edgecolor='black',
                                   linewidth=2)
        ax.add_patch(output_box)
        ax.text(8.25, 7.5, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(8.25, 6.8, task['output'], ha='center', va='center', fontsize=8)
        
        # Arrows
        arrow1 = FancyArrowPatch((3, 7.5), (3.75, 7.5),
                               connectionstyle="arc3,rad=0",
                               arrowstyle="->",
                               mutation_scale=20,
                               color='black')
        ax.add_patch(arrow1)
        
        arrow2 = FancyArrowPatch((6.25, 7.5), (7, 7.5),
                               connectionstyle="arc3,rad=0",
                               arrowstyle="->",
                               mutation_scale=20,
                               color='black')
        ax.add_patch(arrow2)
        
        # Add dimension legend
        ax.text(5, 4, 'H: Height, W: Width\nB: Bands, T: Time\nC: Classes, P: Parameters\nN: Number of objects',
                ha='center', va='center', fontsize=7, style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

def create_ml_suitability_matrix():
    """Create a visual matrix showing ML/DL/FM suitability for different tasks."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define tasks and approaches
    tasks = [
        'Image Classification',
        'Pixel Classification', 
        'Object Detection',
        'Instance Segmentation',
        'Temporal Prediction',
        'Spatial Interpolation',
        'Biophysical Regression'
    ]
    
    approaches = ['Traditional ML', 'Deep Learning', 'Foundation Models']
    
    # Suitability scores (0-3: 0=poor, 1=limited, 2=good, 3=excellent)
    scores = np.array([
        [2, 3, 3],  # Image Classification
        [2, 3, 3],  # Pixel Classification
        [1, 3, 3],  # Object Detection
        [0, 3, 3],  # Instance Segmentation
        [2, 3, 3],  # Temporal Prediction
        [2, 3, 3],  # Spatial Interpolation
        [2, 3, 3],  # Biophysical Regression
    ])
    
    # Create heatmap
    im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=3)
    
    # Set ticks
    ax.set_xticks(np.arange(len(approaches)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(approaches)
    ax.set_yticklabels(tasks)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Suitability', rotation=90, va="bottom")
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Poor', 'Limited', 'Good', 'Excellent'])
    
    # Add text annotations
    suitability_text = [
        ['Limited data\nInterpretable', 'Large data\nHigh accuracy', 'Few-shot\nGeneral features'],
        ['Per-pixel RF\nSVM', 'U-Net\nDeepLab', 'SAM\n+ prompting'],
        ['HOG+SVM\nLimited', 'YOLO\nFaster R-CNN', 'DINO\nOWL-ViT'],
        ['Very limited', 'Mask R-CNN', 'SAM\nOneFormer'],
        ['ARIMA\nRF', 'LSTM\nTCN', 'TimesFM\nPrithvi'],
        ['Kriging\nIDW', 'CNN AE', 'MAE models'],
        ['RF\nSVR', 'CNN\nViT', 'Prithvi\nSatMAE']
    ]
    
    # Add text to cells
    for i in range(len(tasks)):
        for j in range(len(approaches)):
            text = ax.text(j, i, suitability_text[i][j],
                         ha="center", va="center", color="black" if scores[i, j] > 1.5 else "white",
                         fontsize=8)
    
    ax.set_title("ML/DL/FM Suitability Matrix for Geospatial Tasks", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Approach", fontsize=12)
    ax.set_ylabel("Task Type", fontsize=12)
    
    # Add grid
    ax.set_xticks(np.arange(len(approaches)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(tasks)+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    return fig

def save_all_figures():
    """Generate and save all visualization figures."""
    # Create figures
    fig1 = create_prediction_hierarchy_diagram()
    fig2 = create_input_output_diagram()
    fig3 = create_ml_suitability_matrix()
    
    # Save figures
    fig1.savefig('book/extras/images/prediction_hierarchy_overview.png', dpi=300, bbox_inches='tight')
    fig2.savefig('book/extras/images/input_output_relationships.png', dpi=300, bbox_inches='tight')
    fig3.savefig('book/extras/images/ml_suitability_matrix.png', dpi=300, bbox_inches='tight')
    
    print("Figures saved successfully!")
    
    # Close figures to free memory
    plt.close('all')

if __name__ == "__main__":
    save_all_figures()
