import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle
import pandas as pd

def create_xg_heatmap(xg_model, figsize=(5, 7), resolution=100, 
                      shot_technique='Normal', shot_body_part='Right Foot'):
    """
    Create an xG heatmap showing probability of scoring from different pitch positions.
    
    Parameters:
    -----------
    xg_model : SimpleXGModel
        Trained xG model
    figsize : tuple
        Figure size (width, height)
    resolution : int
        Grid resolution (higher = smoother but slower)
    shot_technique : str
        Default shot technique to use for prediction
    shot_body_part : str
        Default body part to use for prediction
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    
    if not xg_model.is_trained:
        raise ValueError("Model must be trained before creating heatmap")
    
    # Create grid of positions
    x_range = np.linspace(60, 120, resolution)  # Only attacking half
    y_range = np.linspace(0, 80, resolution)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Flatten for predictions
    positions = np.c_[X_grid.ravel(), Y_grid.ravel()]
    
    # Calculate distance and angle for each position
    goal_x, goal_y = 120, 40
    dx = goal_x - positions[:, 0]
    dy = goal_y - positions[:, 1]
    
    distances = np.sqrt(dx**2 + dy**2)
    angles = np.arctan(7.32 * dx / (dx**2 + dy**2 - (7.32/2)**2))
    angles = np.abs(angles)
    
    # Create dataframe for predictions
    prediction_df = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'dist_to_goal': distances,
        'angle_to_goal_rad': angles,
        'shot_technique': shot_technique,
        'shot_body_part': shot_body_part
    })
    
    # Get xG predictions
    xg_values = xg_model.predict_xg(prediction_df)
    xg_grid = xg_values.reshape(X_grid.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(60, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect('equal')
    
    # Plot heatmap
    contour = ax.contourf(X_grid, Y_grid, xg_grid, levels=20, cmap='RdYlBu_r', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('Expected Goals (xG)', rotation=270, labelpad=20, fontsize=6)
    
    # Draw pitch elements (half pitch only)
    # Pitch outline
    ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], color='black', linewidth=2)
    
    # Center line
    ax.plot([60, 60], [0, 80], color='black', linewidth=2)
    
    # Center circle (half)
    center_circle = Arc((60, 40), 18.3, 18.3, angle=0, theta1=270, theta2=90, 
                        color='black', linewidth=2, fill=False)
    ax.add_patch(center_circle)
    
    # Penalty area
    penalty_box = Rectangle((102, 18), 18, 44, fill=False, 
                            edgecolor='black', linewidth=2)
    ax.add_patch(penalty_box)
    
    # 6-yard box
    six_yard_box = Rectangle((114, 30), 6, 20, fill=False, 
                             edgecolor='black', linewidth=2)
    ax.add_patch(six_yard_box)
    
    # Penalty arc
    penalty_arc = Arc((108, 40), 18.3, 18.3, angle=0, theta1=130, theta2=230, 
                     color='black', linewidth=2, fill=False)
    ax.add_patch(penalty_arc)
    
    # Penalty spot
    ax.scatter(108, 40, color='black', s=30, zorder=5)
    
    # Goal
    ax.plot([120, 120], [36, 44], color='black', linewidth=5)
    
    # Styling
    ax.set_facecolor('#2d7a2d')
    ax.axis('off')
    
    plt.title(f'Expected Goals (xG) Heatmap\nTechnique: {shot_technique}, Body Part: {shot_body_part}', 
              fontsize=7, weight='bold', pad=20)
    plt.tight_layout()
    
    return fig, ax


def create_xg_heatmap_with_shots(xg_model, shots_df, figsize=(5, 7), 
                                  resolution=100, max_shots=None,
                                  shot_technique='Normal', shot_body_part='Right Foot'):
    """
    Create xG heatmap with actual shots overlaid.
    
    Parameters:
    -----------
    xg_model : SimpleXGModel
        Trained xG model
    shots_df : pd.DataFrame
        DataFrame with actual shots
    figsize : tuple
        Figure size
    resolution : int
        Grid resolution
    max_shots : int
        Maximum number of shots to display
    shot_technique : str
        Default shot technique for heatmap
    shot_body_part : str
        Default body part for heatmap
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    
    # Create base heatmap
    fig, ax = create_xg_heatmap(xg_model, figsize, resolution, 
                                shot_technique, shot_body_part)
    
    # Filter to attacking half
    shots_attacking = shots_df[shots_df['x'] >= 60].copy()
    
    if max_shots:
        shots_attacking = shots_attacking.head(max_shots)
    
    # Color mapping for outcomes
    outcome_colors = {
        'Goal': '#00FF00',
        'Saved': '#FFA500',
        'Off T': '#FF0000',
        'Blocked': '#FFFF00',
        'Wayward': '#FF69B4',
        'Post': '#00FFFF',
    }
    
    # Plot shots
    for idx, shot in shots_attacking.iterrows():
        color = outcome_colors.get(shot.get('shot_outcome', ''), 'white')
        
        # Larger marker for goals
        size = 150 if shot.get('shot_outcome') == 'Goal' else 80
        
        ax.scatter(shot['x'], shot['y'], 
                  c=color, s=size, alpha=0.8,
                  edgecolors='black', linewidth=2,
                  zorder=10)
    
    return fig, ax


def compare_xg_by_technique(xg_model, techniques=None, figsize=(7, 5)):
    """
    Create side-by-side xG heatmaps for different shot techniques.
    
    Parameters:
    -----------
    xg_model : SimpleXGModel
        Trained xG model
    techniques : list
        List of techniques to compare (None = use common ones)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    
    if techniques is None:
        techniques = ['Normal', 'Half Volley', 'Volley', 'Header']
    
    n_techniques = len(techniques)
    fig, axes = plt.subplots(1, n_techniques, figsize=figsize)
    
    if n_techniques == 1:
        axes = [axes]
    
    resolution = 80  # Lower resolution for multiple plots
    
    for idx, technique in enumerate(techniques):
        ax = axes[idx]
        
        # Create grid
        x_range = np.linspace(60, 120, resolution)
        y_range = np.linspace(0, 80, resolution)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        
        positions = np.c_[X_grid.ravel(), Y_grid.ravel()]
        
        # Calculate features
        goal_x, goal_y = 120, 40
        dx = goal_x - positions[:, 0]
        dy = goal_y - positions[:, 1]
        distances = np.sqrt(dx**2 + dy**2)
        angles = np.arctan(7.32 * dx / (dx**2 + dy**2 - (7.32/2)**2))
        angles = np.abs(angles)
        
        prediction_df = pd.DataFrame({
            'x': positions[:, 0],
            'y': positions[:, 1],
            'dist_to_goal': distances,
            'angle_to_goal_rad': angles,
            'shot_technique': technique,
            'shot_body_part': 'Right Foot'
        })
        
        xg_values = xg_model.predict_xg(prediction_df)
        xg_grid = xg_values.reshape(X_grid.shape)
        
        # Plot
        ax.set_xlim(60, 120)
        ax.set_ylim(0, 80)
        ax.set_aspect('equal')
        
        contour = ax.contourf(X_grid, Y_grid, xg_grid, levels=20, 
                             cmap='RdYlBu_r', alpha=0.8, vmin=0, vmax=1)
        
        # Draw pitch
        ax.plot([60, 120, 120, 60, 60], [0, 0, 80, 80, 0], 
               color='black', linewidth=2)
        ax.plot([60, 60], [0, 80], color='black', linewidth=2)
        
        penalty_box = Rectangle((102, 18), 18, 44, fill=False, 
                               edgecolor='black', linewidth=2)
        ax.add_patch(penalty_box)
        
        ax.plot([120, 120], [36, 44], color='black', linewidth=5)
        
        ax.set_facecolor('#2d7a2d')
        ax.axis('off')
        ax.set_title(technique, fontsize=6, weight='bold')
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('xG', rotation=270, labelpad=20, fontsize=6)
    
    plt.suptitle('xG Heatmap Comparison by Shot Technique', 
                fontsize=8, weight='bold', y=0.98)
    
    return fig, axes
