import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from loguru import logger

# Now create heatmaps using technique-specific models
def create_xg_heatmap_grid(x_bins=20, y_bins=13):
    """Create a grid for xG heatmap."""
    x_edges = np.linspace(0, 105, x_bins + 1)
    y_edges = np.linspace(0, 68, y_bins + 1)
    return x_edges, y_edges

def generate_xg_heatmap(model, df, technique, x_bins=20, y_bins=13):
    """Generate xG heatmap for a specific technique using its model."""
    x_edges, y_edges = create_xg_heatmap_grid(x_bins, y_bins)
    
    # Create grid
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    X_grid, Y_grid = np.meshgrid(x_centers, y_centers)
    
    # Create dataframe for prediction
    grid_df = pd.DataFrame({
        'x': X_grid.ravel(),
        'y': Y_grid.ravel()
    })
    
    # Calculate distance and angle
    grid_df['dist_to_goal'] = np.sqrt((105 - grid_df['x'])**2 + (34 - grid_df['y'])**2)
    grid_df['angle_to_goal_rad'] = np.arctan2(
        abs(grid_df['y'] - 34),
        105 - grid_df['x']
    )
    
    # Add default shot_body_part if needed
    if 'shot_body_part' in df.columns:
        # Use most common body part for this technique
        technique_data = df[df['shot_technique'] == technique]
        most_common_bodypart = technique_data['shot_body_part'].mode()[0]
        grid_df['shot_body_part'] = most_common_bodypart
    
    # Predict xG using technique-specific model
    xg_values = model.predict_xg(grid_df)
    
    # Reshape to grid
    xg_grid = xg_values.reshape(y_bins, x_bins)
    
    return xg_grid, x_edges, 

def generate_xg_heatmap_from_data(df, technique, x_bins=20, y_bins=13):
    """Generate xG heatmap using actual data range."""
    technique_data = df[df['shot_technique'] == technique].copy()
    
    if len(technique_data) == 0:
        return None, None, None
    
    # Use actual data range from the full dataset
    result = binned_statistic_2d(
        technique_data['x'], 
        technique_data['y'],
        technique_data['xG'],
        statistic='mean',
        bins=[x_bins, y_bins],
        range=[[40, 120], [0, 80]]  # Updated to match your data!
    )
    
    xg_grid = result.statistic.T
    x_edges = result.x_edge
    y_edges = result.y_edge
    
    return xg_grid, x_edges, y_edges


def verify_xg_heatmaps(df):
    """
    Comprehensive verification of xG heatmap data and binning.
    """
    logger.info("=" * 80)
    logger.info("XG HEATMAP VERIFICATION REPORT")
    logger.info("=" * 80)
    
    # 1. Overall Dataset Check
    logger.info("\n1. OVERALL DATASET")
    logger.info("-" * 80)
    logger.info(f"Total shots: {len(df):,}")
    logger.info(f"Shots with xG values: {df['xG'].notna().sum():,}")
    logger.info(f"Missing xG values: {df['xG'].isna().sum()}")
    logger.info(f"\nxG Statistics:")
    logger.info(f"  Mean xG: {df['xG'].mean():.4f}")
    logger.info(f"  Median xG: {df['xG'].median():.4f}")
    logger.info(f"  Min xG: {df['xG'].min():.4f}")
    logger.info(f"  Max xG: {df['xG'].max():.4f}")
    
    # 2. Coordinate Range Check
    logger.info("\n2. COORDINATE RANGES")
    logger.info("-" * 80)
    logger.info(f"X range: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
    logger.info(f"Y range: [{df['y'].min():.1f}, {df['y'].max():.1f}]")
    logger.info(f"Distance range: [{df['dist_to_goal'].min():.1f}, {df['dist_to_goal'].max():.1f}]")
    
    # Check if any shots fall outside expected range
    outside_x = df[(df['x'] < 40) | (df['x'] > 120)]
    outside_y = df[(df['y'] < 0) | (df['y'] > 80)]
    logger.info(f"\nShots outside [40, 120] x [0, 80] grid:")
    logger.info(f"  Outside X: {len(outside_x):,}")
    logger.info(f"  Outside Y: {len(outside_y):,}")
    
    # 3. Technique Distribution
    logger.info("\n3. TECHNIQUE DISTRIBUTION")
    logger.info("-" * 80)
    technique_counts = df['shot_technique'].value_counts().sort_values(ascending=False)
    for technique, count in technique_counts.items():
        pct = 100 * count / len(df)
        logger.info(f"  {technique:20s}: {count:6,} shots ({pct:5.2f}%)")
    
    # 4. Per-Technique Statistics
    logger.info("\n4. PER-TECHNIQUE XG STATISTICS")
    logger.info("-" * 80)
    logger.info(f"{'Technique':<20} {'Count':>8} {'Mean xG':>10} {'Median xG':>10} {'Max xG':>10}")
    logger.info("-" * 80)
    for technique in technique_counts.index:
        tech_data = df[df['shot_technique'] == technique]
        logger.info(f"{technique:<20} {len(tech_data):>8,} "
              f"{tech_data['xG'].mean():>10.4f} "
              f"{tech_data['xG'].median():>10.4f} "
              f"{tech_data['xG'].max():>10.4f}")
    
    # 5. Binning Test
    logger.info("\n5. BINNING TEST (20x13 grid)")
    logger.info("-" * 80)
    
    from scipy.stats import binned_statistic_2d
    
    total_binned = 0
    for technique in technique_counts.index:
        tech_data = df[df['shot_technique'] == technique]
        
        result = binned_statistic_2d(
            tech_data['x'], 
            tech_data['y'],
            tech_data['xG'],
            statistic='count',
            bins=[20, 13],
            range=[[40, 120], [0, 80]]
        )
        
        count_grid = result.statistic.T
        shots_binned = int(np.nansum(count_grid))
        bins_filled = np.sum(count_grid > 0)
        coverage = 100 * shots_binned / len(tech_data)
        
        total_binned += shots_binned
        
        logger.info(f"{technique:<20}: {shots_binned:>6}/{len(tech_data):>6} binned "
              f"({coverage:>5.1f}%), {bins_filled:>3}/260 bins filled")
    
    logger.info(f"{'TOTAL':<20}: {total_binned:>6}/{len(df):>6} binned")
    
    # 6. Distance Distribution by Technique
    logger.info("\n6. DISTANCE DISTRIBUTION BY TECHNIQUE")
    logger.info("-" * 80)
    logger.info(f"{'Technique':<20} {'<10m':>7} {'10-20m':>7} {'20-30m':>7} {'>30m':>7}")
    logger.info("-" * 80)
    for technique in technique_counts.index:
        tech_data = df[df['shot_technique'] == technique]
        d = tech_data['dist_to_goal']
        
        under_10 = (d < 10).sum()
        m10_20 = ((d >= 10) & (d < 20)).sum()
        m20_30 = ((d >= 20) & (d < 30)).sum()
        over_30 = (d >= 30).sum()
        
        logger.info(f"{technique:<20} {under_10:>7} {m10_20:>7} {m20_30:>7} {over_30:>7}")
    
    # 7. Goal Conversion by Technique
    if 'shot_outcome' in df.columns:
        logger.info("\n7. GOAL CONVERSION RATES")
        logger.info("-" * 80)
        logger.info(f"{'Technique':<20} {'Goals':>7} {'Shots':>7} {'Conv%':>7} {'Avg xG':>9}")
        logger.info("-" * 80)
        for technique in technique_counts.index:
            tech_data = df[df['shot_technique'] == technique]
            goals = (tech_data['shot_outcome'] == 'Goal').sum()
            shots = len(tech_data)
            conv_rate = 100 * goals / shots
            avg_xg = tech_data['xG'].mean()
            
            logger.info(f"{technique:<20} {goals:>7} {shots:>7} {conv_rate:>6.2f}% {avg_xg:>9.4f}")
    
    # 8. Data Quality Check
    logger.info("\n8. DATA QUALITY CHECKS")
    logger.info("-" * 80)
    
    # Check for suspicious values
    suspicious = []
    
    if (df['xG'] < 0).any():
        suspicious.append(f"⚠️  Negative xG values: {(df['xG'] < 0).sum()}")
    if (df['xG'] > 1).any():
        suspicious.append(f"⚠️  xG > 1: {(df['xG'] > 1).sum()}")
    if df['xG'].isna().any():
        suspicious.append(f"⚠️  Missing xG: {df['xG'].isna().sum()}")
    if (df['dist_to_goal'] < 0).any():
        suspicious.append(f"⚠️  Negative distance: {(df['dist_to_goal'] < 0).sum()}")
    if (df['dist_to_goal'] > 120).any():
        suspicious.append(f"⚠️  Distance > 120m: {(df['dist_to_goal'] > 120).sum()}")
    
    if suspicious:
        for issue in suspicious:
            logger.info(issue)
    else:
        logger.info("✅ All data quality checks passed!")
    
    # 9. Visual Binning Example
    logger.info("\n9. SAMPLE BINNING FOR 'NORMAL' SHOTS")
    logger.info("-" * 80)
    
    normal_data = df[df['shot_technique'] == 'Normal']
    
    result = binned_statistic_2d(
        normal_data['x'], 
        normal_data['y'],
        normal_data['xG'],
        statistic='mean',
        bins=[20, 13],
        range=[[40, 120], [0, 80]]
    )
    
    xg_grid = result.statistic.T
    
    logger.info(f"Grid shape: {xg_grid.shape}")
    logger.info(f"Bins with data: {np.sum(~np.isnan(xg_grid))}/260")
    logger.info(f"xG range in bins: [{np.nanmin(xg_grid):.4f}, {np.nanmax(xg_grid):.4f}]")
    logger.info(f"Mean xG across bins: {np.nanmean(xg_grid):.4f}")
    
    # Show a slice of the grid (bins closest to goal)
    logger.info("\nSample: Last 5 X bins (closest to goal), all Y bins:")
    logger.info("(Rows = Y position, Cols = X position)")
    sample = xg_grid[:, -5:]
    logger.info.info(pd.DataFrame(sample).to_string(float_format='%.3f'))
    
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 80)


def plot_distance_comparison(df):
    """Plot distance distributions for different techniques."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    techniques = df['shot_technique'].unique()
    
    for idx, technique in enumerate(techniques):
        tech_data = df[df['shot_technique'] == technique]
        
        ax = axes[idx]
        ax.hist(tech_data['dist_to_goal'], bins=30, alpha=0.7, 
                edgecolor='black', color='steelblue')
        ax.axvline(tech_data['dist_to_goal'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {tech_data['dist_to_goal'].mean():.1f}m")
        ax.set_xlabel('Distance to Goal (m)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'{technique} (n={len(tech_data)})', fontsize=13, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    for idx in range(len(techniques), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Distance Distribution by Technique', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()
