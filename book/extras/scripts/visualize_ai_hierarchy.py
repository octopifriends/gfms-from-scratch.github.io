"""
Visualization script for AI/ML/DL/FM hierarchy in geospatial context.
Creates diagrams to accompany the hierarchy explainer.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.lines as mlines

def create_nested_hierarchy_diagram():
    """Create a visual representation of the AI/ML/DL/FM nested hierarchy."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Define nested boxes with labels and examples
    boxes = [
        {
            'name': 'Artificial Intelligence (AI)',
            'subtitle': 'Any computational algorithm',
            'rect': (1, 1, 10, 8),
            'color': '#e3f2fd',
            'examples': ['Rule-based systems', 'Expert systems', 'Geometric algorithms']
        },
        {
            'name': 'Machine Learning (ML)',
            'subtitle': 'Algorithms that learn from data',
            'rect': (2, 2, 8, 6),
            'color': '#bbdefb',
            'examples': ['Random Forest', 'SVM', 'k-means']
        },
        {
            'name': 'Deep Learning (DL)',
            'subtitle': 'Multi-layer neural networks',
            'rect': (3, 3, 6, 4),
            'color': '#90caf9',
            'examples': ['CNN', 'U-Net', 'LSTM']
        },
        {
            'name': 'Foundation Models (FM)',
            'subtitle': 'Large pre-trained models',
            'rect': (4, 4, 4, 2),
            'color': '#64b5f6',
            'examples': ['Prithvi', 'SatMAE', 'Clay']
        }
    ]
    
    # Draw boxes from outermost to innermost
    for box in boxes:
        x, y, w, h = box['rect']
        
        # Main box
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=box['color'],
                              edgecolor='black',
                              linewidth=2)
        ax.add_patch(rect)
        
        # Title
        ax.text(x + w/2, y + h - 0.3, box['name'],
                ha='center', va='top', fontsize=14, weight='bold')
        
        # Subtitle
        ax.text(x + w/2, y + h - 0.6, box['subtitle'],
                ha='center', va='top', fontsize=10, style='italic')
        
        # Examples
        example_text = ' • '.join(box['examples'])
        ax.text(x + w/2, y + 0.3, example_text,
                ha='center', va='bottom', fontsize=9, color='#333')
    
    # Add arrows showing the "subset of" relationship
    arrow_props = dict(arrowstyle='->', lw=2, color='#666')
    
    # Add title
    ax.text(6, 9.5, 'The AI Hierarchy in Geospatial Science',
            ha='center', fontsize=18, weight='bold')
    
    # Add note
    ax.text(6, 0.3, 'Each inner category is a specialized subset of the outer category',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat'))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_timeline_evolution():
    """Create a timeline showing the evolution of approaches in remote sensing."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Timeline data
    eras = [
        {
            'period': '1970s-1990s',
            'name': 'Rule-Based Era',
            'y': 4,
            'color': '#e3f2fd',
            'methods': ['Band ratios', 'Threshold rules', 'Expert systems']
        },
        {
            'period': '1990s-2010s',
            'name': 'Classical ML Era',
            'y': 3,
            'color': '#bbdefb',
            'methods': ['Maximum likelihood', 'Decision trees', 'SVM']
        },
        {
            'period': '2010s-2020s',
            'name': 'Deep Learning Era',
            'y': 2,
            'color': '#90caf9',
            'methods': ['CNNs', 'Transfer learning', 'Segmentation nets']
        },
        {
            'period': '2020s-Present',
            'name': 'Foundation Model Era',
            'y': 1,
            'color': '#64b5f6',
            'methods': ['Self-supervised', 'Multi-modal', 'Few-shot learning']
        }
    ]
    
    # Draw timeline base
    ax.axhline(y=2.5, color='black', linewidth=2, zorder=1)
    
    # Draw era boxes
    x_positions = np.linspace(2, 12, len(eras))
    
    for i, (era, x) in enumerate(zip(eras, x_positions)):
        # Era box
        box = FancyBboxPatch((x-1.2, era['y']-0.4), 2.4, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=era['color'],
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(box)
        
        # Era name and period
        ax.text(x, era['y'], era['name'],
                ha='center', va='center', fontsize=12, weight='bold')
        ax.text(x, era['y']-0.7, era['period'],
                ha='center', va='top', fontsize=10)
        
        # Methods
        for j, method in enumerate(era['methods']):
            ax.text(x, era['y']-1.2-j*0.3, f"• {method}",
                    ha='center', va='top', fontsize=9, color='#444')
        
        # Connect to timeline
        ax.plot([x, x], [era['y']-0.4, 2.5], 'k--', alpha=0.5)
    
    # Add title
    ax.text(7, 5.5, 'Evolution of Geospatial AI Approaches',
            ha='center', fontsize=18, weight='bold')
    
    # Add trend arrow
    ax.annotate('', xy=(12.5, 2.5), xytext=(1.5, 2.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='#333'))
    ax.text(7, 2.2, 'Increasing Complexity & Capability',
            ha='center', fontsize=11, style='italic')
    
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_comparison_matrix():
    """Create a comparison matrix of different AI levels."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Categories and criteria
    categories = ['AI/Rule-Based', 'Classical ML', 'Deep Learning', 'Foundation Models']
    criteria = [
        'Data Efficiency',  # Changed from Data Requirements (inverted)
        'Computational Efficiency',  # Changed from Computational Cost (inverted)
        'Interpretability',
        'Accuracy Potential',
        'Generalization',
        'Development Speed'  # Changed from Development Time (inverted)
    ]
    
    # Scores (1-5 scale) - now higher is always better
    scores = np.array([
        [5, 4, 5, 3, 2, 5],  # AI/Rule-Based
        [3, 3, 4, 3, 3, 3],  # Classical ML
        [1, 2, 2, 4, 4, 2],  # Deep Learning
        [4, 1, 1, 5, 5, 4],  # Foundation Models (less data needed for fine-tuning)
    ])
    
    # Color map - now green is good, red is bad
    colors = plt.cm.RdYlGn
    
    # Create matrix
    for i, category in enumerate(categories):
        for j, criterion in enumerate(criteria):
            # Draw cell
            color = colors(scores[i, j] / 5.0)
            rect = Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add score
            ax.text(j + 0.5, i + 0.5, str(scores[i, j]),
                    ha='center', va='center', fontsize=12, weight='bold')
    
    # Add labels
    for i, category in enumerate(categories):
        ax.text(-0.1, i + 0.5, category,
                ha='right', va='center', fontsize=11, weight='bold')
    
    for j, criterion in enumerate(criteria):
        ax.text(j + 0.5, -0.1, criterion,
                ha='center', va='top', fontsize=11, weight='bold', rotation=45)
    
    # Add title
    ax.text(3, 4.5, 'Comparison Matrix: Computational Hierarchy Levels',
            ha='center', fontsize=16, weight='bold')
    
    # Add legend
    legend_elements = [
        mlines.Line2D([0], [0], marker='s', color='w', 
                     markerfacecolor=colors(1/5), markersize=15, label='1 = Poor'),
        mlines.Line2D([0], [0], marker='s', color='w',
                     markerfacecolor=colors(3/5), markersize=15, label='3 = Moderate'),
        mlines.Line2D([0], [0], marker='s', color='w',
                     markerfacecolor=colors(5/5), markersize=15, label='5 = Excellent')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_geospatial_examples_chart():
    """Create a chart showing specific geospatial examples at each level."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Geospatial Applications at Each AI Level', fontsize=18, weight='bold')
    
    # AI/Rule-Based
    ax = axes[0, 0]
    ax.set_title('AI: Rule-Based Systems', fontsize=14, weight='bold')
    
    examples = [
        ('NDVI Threshold', 'if NDVI > 0.3: vegetation'),
        ('Water Detection', 'if NDWI > 0: water'),
        ('Slope Analysis', 'if slope > 30°: steep'),
        ('Buffer Zones', 'if distance < 100m: buffer')
    ]
    
    for i, (name, rule) in enumerate(examples):
        y = 0.8 - i * 0.2
        ax.text(0.1, y, f"• {name}:", fontsize=11, weight='bold')
        ax.text(0.15, y - 0.05, rule, fontsize=10, style='italic', color='#555')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Classical ML
    ax = axes[0, 1]
    ax.set_title('ML: Learning from Features', fontsize=14, weight='bold')
    
    # Simple decision boundary visualization
    np.random.seed(42)
    n_points = 50
    
    # Class 1: Forest (high NDVI, low brightness)
    forest_ndvi = np.random.normal(0.7, 0.1, n_points)
    forest_bright = np.random.normal(0.3, 0.1, n_points)
    
    # Class 2: Urban (low NDVI, high brightness)
    urban_ndvi = np.random.normal(0.2, 0.1, n_points)
    urban_bright = np.random.normal(0.7, 0.1, n_points)
    
    ax.scatter(forest_ndvi, forest_bright, c='green', label='Forest', alpha=0.6)
    ax.scatter(urban_ndvi, urban_bright, c='gray', label='Urban', alpha=0.6)
    
    # Decision boundary
    x = np.linspace(0, 1, 100)
    y = -x + 1
    ax.plot(x, y, 'r--', label='SVM boundary')
    
    ax.set_xlabel('NDVI')
    ax.set_ylabel('Brightness')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Deep Learning
    ax = axes[1, 0]
    ax.set_title('DL: End-to-End Learning', fontsize=14, weight='bold')
    
    # Show CNN architecture sketch
    layers = [
        ('Input\n13 bands', 0.1, 0.8, 0.1),
        ('Conv1\n64 filters', 0.3, 0.7, 0.15),
        ('Conv2\n128 filters', 0.5, 0.6, 0.2),
        ('Conv3\n256 filters', 0.7, 0.5, 0.25),
        ('Output\n10 classes', 0.9, 0.4, 0.1)
    ]
    
    for i, (name, x, h, w) in enumerate(layers):
        rect = Rectangle((x - w/2, 0.2), w, h, 
                        facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, 0.1, name, ha='center', fontsize=9)
        
        if i < len(layers) - 1:
            ax.arrow(x + w/2, 0.5, 0.15, 0, head_width=0.05, 
                    head_length=0.02, fc='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Foundation Models
    ax = axes[1, 1]
    ax.set_title('FM: Pre-train Once, Use Many', fontsize=14, weight='bold')
    
    # Show transfer learning concept
    # Pre-trained model
    rect = FancyBboxPatch((0.1, 0.6), 0.3, 0.3,
                         boxstyle="round,pad=0.02",
                         facecolor='gold', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.25, 0.75, 'Pre-trained\nPrithvi', ha='center', va='center', 
            fontsize=11, weight='bold')
    
    # Multiple downstream tasks
    tasks = [
        ('Flood\nMapping', 0.6, 0.8),
        ('Crop\nClassification', 0.6, 0.6),
        ('Change\nDetection', 0.6, 0.4),
        ('Cloud\nRemoval', 0.6, 0.2)
    ]
    
    for task, x, y in tasks:
        rect = Rectangle((x - 0.08, y - 0.05), 0.16, 0.1,
                        facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, task, ha='center', va='center', fontsize=9)
        ax.arrow(0.4, 0.75, x - 0.45, y - 0.75, head_width=0.03,
                head_length=0.02, fc='gray', alpha=0.5)
    
    ax.text(0.5, 0.05, 'One model → Many applications',
            ha='center', fontsize=10, style='italic')
    
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import os
    
    output_dir = "../images/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create diagrams
    fig1 = create_nested_hierarchy_diagram()
    fig1.savefig(os.path.join(output_dir, "ai_hierarchy_nested.png"), 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    fig2 = create_timeline_evolution()
    fig2.savefig(os.path.join(output_dir, "ai_timeline_evolution.png"), 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    fig3 = create_comparison_matrix()
    fig3.savefig(os.path.join(output_dir, "ai_comparison_matrix.png"), 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    
    fig4 = create_geospatial_examples_chart()
    fig4.savefig(os.path.join(output_dir, "ai_geospatial_examples.png"), 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    
    print("AI hierarchy diagrams created successfully!")
    print(f"- {output_dir}ai_hierarchy_nested.png")
    print(f"- {output_dir}ai_timeline_evolution.png")
    print(f"- {output_dir}ai_comparison_matrix.png")
    print(f"- {output_dir}ai_geospatial_examples.png")
