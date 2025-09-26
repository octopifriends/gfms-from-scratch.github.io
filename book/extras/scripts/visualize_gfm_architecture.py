"""
Visualization script for Geospatial Foundation Model architecture.
Creates diagrams to accompany the GFM predictions explainer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

def create_gfm_architecture_diagram():
    """Create a visual representation of the GFM architecture flow."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define positions
    input_x, input_y = 1, 4
    encoder_x, encoder_y = 4, 4
    embedding_x, embedding_y = 7, 4
    decoder_x, decoder_y = 10, 5
    alt_x, alt_y = 10, 3
    
    # Draw components
    # Input data cube
    ax.add_patch(Rectangle((input_x-0.4, input_y-0.3), 0.8, 0.6, 
                          facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(input_x, input_y+0.5, 'Satellite Data\n3×100×100×12', 
            ha='center', va='bottom', fontsize=10, weight='bold')
    ax.text(input_x, input_y-0.5, 'RGB × Spatial × Time', 
            ha='center', va='top', fontsize=8, style='italic')
    
    # Encoder
    ax.add_patch(FancyBboxPatch((encoder_x-0.5, encoder_y-0.4), 1, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(encoder_x, encoder_y, 'Encoder\n(Transformer)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Embedding
    ax.add_patch(Rectangle((embedding_x-0.3, embedding_y-0.4), 0.6, 0.8,
                          facecolor='yellow', edgecolor='black', linewidth=2))
    ax.text(embedding_x, embedding_y, 'Embedding\nVector', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Decoder branch
    ax.add_patch(FancyBboxPatch((decoder_x-0.5, decoder_y-0.4), 1, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax.text(decoder_x, decoder_y, 'Decoder\n(Task-specific)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Alternative ML branch
    ax.add_patch(FancyBboxPatch((alt_x-0.5, alt_y-0.4), 1, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor='lightsalmon', edgecolor='black', linewidth=2))
    ax.text(alt_x, alt_y, 'Traditional ML\n(e.g., Regression)', 
            ha='center', va='center', fontsize=9)
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    ax.annotate('', xy=(encoder_x-0.5, encoder_y), xytext=(input_x+0.4, input_y),
                arrowprops=arrow_props)
    ax.annotate('', xy=(embedding_x-0.3, embedding_y), xytext=(encoder_x+0.5, encoder_y),
                arrowprops=arrow_props)
    ax.annotate('', xy=(decoder_x-0.5, decoder_y), xytext=(embedding_x+0.3, embedding_y+0.2),
                arrowprops=arrow_props)
    ax.annotate('', xy=(alt_x-0.5, alt_y), xytext=(embedding_x+0.3, embedding_y-0.2),
                arrowprops=arrow_props)
    
    # Add outputs
    outputs = [
        (12.5, 5.5, 'Land Cover\nMap'),
        (12.5, 4.5, 'Change\nDetection'),
        (12.5, 3.5, 'Crop Type\nClassification'),
        (12.5, 2.5, 'Biomass\nEstimation')
    ]
    
    for x, y, text in outputs:
        ax.add_patch(Rectangle((x-0.4, y-0.3), 0.8, 0.6,
                              facecolor='lightgray', edgecolor='black', linewidth=1))
        ax.text(x, y, text, ha='center', va='center', fontsize=8)
        ax.annotate('', xy=(x-0.4, y), xytext=(decoder_x+0.5, decoder_y-0.1),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
    
    # Add title and labels
    ax.text(7, 6.5, 'Geospatial Foundation Model Architecture', 
            ha='center', fontsize=16, weight='bold')
    
    ax.text(2.5, 1.5, 'Pre-training Task:\n"Predict missing patches"', 
            ha='center', fontsize=10, style='italic', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    # Set axis properties
    ax.set_xlim(0, 14)
    ax.set_ylim(1, 7)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_pretraining_tasks_diagram():
    """Create a diagram showing different pre-training objectives."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Self-Supervised Pre-training Tasks for GFMs', fontsize=16, weight='bold')
    
    # Masked Autoencoding
    ax = axes[0, 0]
    ax.set_title('Masked Autoencoding', fontsize=12, weight='bold')
    
    # Create checkerboard pattern for masking
    grid = np.ones((10, 10, 3))
    mask = np.random.random((10, 10)) > 0.25
    grid[mask] = [0.8, 0.8, 0.8]
    
    ax.imshow(grid)
    ax.text(5, -1, 'Input: 75% masked', ha='center', fontsize=10)
    ax.text(5, 11, 'Task: Reconstruct full image', ha='center', fontsize=10)
    ax.axis('off')
    
    # Temporal Prediction
    ax = axes[0, 1]
    ax.set_title('Temporal Prediction', fontsize=12, weight='bold')
    
    # Show time series
    times = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', '???']
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 6))
    
    for i, (t, c) in enumerate(zip(times[:-1], colors)):
        ax.add_patch(Rectangle((i*1.2, 0), 1, 1, facecolor=c, edgecolor='black'))
        ax.text(i*1.2+0.5, -0.5, t, ha='center', fontsize=9)
    
    # Question mark for prediction
    ax.add_patch(Rectangle((6*1.2, 0), 1, 1, facecolor='white', edgecolor='black', linestyle='--'))
    ax.text(6*1.2+0.5, 0.5, '?', ha='center', va='center', fontsize=20, weight='bold')
    ax.text(6*1.2+0.5, -0.5, 'Jul', ha='center', fontsize=9)
    
    ax.text(3.5, 1.5, 'Task: Predict next time step', ha='center', fontsize=10)
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-1, 2)
    ax.axis('off')
    
    # Multi-modal Alignment
    ax = axes[1, 0]
    ax.set_title('Multi-modal Alignment', fontsize=12, weight='bold')
    
    # Optical and SAR patches
    ax.add_patch(Rectangle((0, 0), 2, 2, facecolor='lightblue', edgecolor='black'))
    ax.text(1, 1, 'Optical\n(Sentinel-2)', ha='center', va='center', fontsize=10)
    
    ax.add_patch(Rectangle((3, 0), 2, 2, facecolor='lightgray', edgecolor='black'))
    ax.text(4, 1, 'SAR\n(Sentinel-1)', ha='center', va='center', fontsize=10)
    
    ax.text(2.5, -0.5, 'Task: Learn same embedding', ha='center', fontsize=10)
    ax.annotate('', xy=(2.5, 0.5), xytext=(2, 1), arrowprops=dict(arrowstyle='<->', lw=2))
    ax.annotate('', xy=(2.5, 0.5), xytext=(3, 1), arrowprops=dict(arrowstyle='<->', lw=2))
    
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-1, 2.5)
    ax.axis('off')
    
    # Contrastive Learning
    ax = axes[1, 1]
    ax.set_title('Contrastive Learning', fontsize=12, weight='bold')
    
    # Show spatial neighbors
    positions = [(1, 1), (2, 1), (1, 0), (2, 0)]  # Nearby
    for x, y in positions:
        ax.add_patch(Rectangle((x-0.4, y-0.4), 0.8, 0.8, 
                              facecolor='lightgreen', edgecolor='black'))
    
    ax.text(1.5, -0.8, 'Similar embeddings', ha='center', fontsize=9, color='green')
    
    # Distant location
    ax.add_patch(Rectangle((4.6, 0.6), 0.8, 0.8, 
                          facecolor='lightcoral', edgecolor='black'))
    ax.text(5, -0.8, 'Different embedding', ha='center', fontsize=9, color='red')
    
    ax.text(3, 2, 'Task: Nearby patches → similar vectors', ha='center', fontsize=10)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(-1, 2.5)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_task_hierarchy_diagram():
    """Create a diagram showing the hierarchy of downstream tasks."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Central embedding
    center_x, center_y = 7, 5
    ax.add_patch(patches.Circle((center_x, center_y), 0.8, 
                               facecolor='gold', edgecolor='black', linewidth=3))
    ax.text(center_x, center_y, 'Pre-trained\nEmbedding', 
            ha='center', va='center', fontsize=12, weight='bold')
    
    # Task categories with examples
    task_categories = [
        {
            'name': 'Pixel-Level Tasks',
            'pos': (3, 8),
            'color': 'lightblue',
            'tasks': ['Land Cover', 'Cloud Masking', 'Change Detection']
        },
        {
            'name': 'Image-Level Tasks',
            'pos': (11, 8),
            'color': 'lightgreen',
            'tasks': ['Scene Classification', 'Yield Estimation', 'Damage Assessment']
        },
        {
            'name': 'Time Series Tasks',
            'pos': (3, 2),
            'color': 'lightcoral',
            'tasks': ['Crop Monitoring', 'Phenology', 'Trend Analysis']
        },
        {
            'name': 'Multi-Modal Tasks',
            'pos': (11, 2),
            'color': 'lightsalmon',
            'tasks': ['Gap Filling', 'Super-Resolution', 'Data Fusion']
        }
    ]
    
    for category in task_categories:
        x, y = category['pos']
        
        # Main category box
        ax.add_patch(FancyBboxPatch((x-1.5, y-0.5), 3, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=category['color'], 
                                   edgecolor='black', linewidth=2))
        ax.text(x, y, category['name'], ha='center', va='center', 
                fontsize=11, weight='bold')
        
        # Draw connection to center
        ax.annotate('', xy=(x, y-0.5 if y > center_y else y+0.5), 
                    xytext=(center_x, center_y),
                    arrowprops=dict(arrowstyle='<-', lw=2, color='gray'))
        
        # Add task examples
        for i, task in enumerate(category['tasks']):
            task_y = y - 1.5 - i*0.6 if y > center_y else y + 1.5 + i*0.6
            ax.text(x, task_y, f'• {task}', ha='center', va='center', fontsize=9)
    
    # Add title
    ax.text(center_x, 10, 'Downstream Tasks Enabled by GFM Pre-training', 
            ha='center', fontsize=16, weight='bold')
    
    # Add annotation
    ax.text(center_x, 0.5, 
            'Each task requires only minimal fine-tuning of the pre-trained model',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create and save diagrams
    import os
    
    output_dir = "../images/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Architecture diagram
    fig1 = create_gfm_architecture_diagram()
    fig1.savefig(os.path.join(output_dir, "gfm_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Pre-training tasks
    fig2 = create_pretraining_tasks_diagram()
    fig2.savefig(os.path.join(output_dir, "gfm_pretraining_tasks.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Task hierarchy
    fig3 = create_task_hierarchy_diagram()
    fig3.savefig(os.path.join(output_dir, "gfm_task_hierarchy.png"), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Diagrams created successfully!")
    print(f"- {output_dir}gfm_architecture.png")
    print(f"- {output_dir}gfm_pretraining_tasks.png")
    print(f"- {output_dir}gfm_task_hierarchy.png")
