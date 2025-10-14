# statsbomb_shots.py
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc
import pandas as pd


# CONFIG: point this to your local copy of StatsBomb open-data
OPEN_DATA_DIR = Path("open-data/data")  # adjust if different
EVENTS_DIR = OPEN_DATA_DIR / "events"

# Geometry for StatsBomb coordinates:
# StatsBomb uses a 120x80 pitch with the attacking goal on the right at x=120, centered y=40
GOAL_X = 120.0
GOAL_CENTER_Y = 40.0
LEFT_POST = (GOAL_X, GOAL_CENTER_Y - 4.0)   # post positions assuming 8 unit goal width (36 and 44)
RIGHT_POST = (GOAL_X, GOAL_CENTER_Y + 4.0)

def load_events_json(path: Path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def shot_distance(x, y):
    return euclid((x,y), (GOAL_X, GOAL_CENTER_Y))

def shot_angle(x, y):
    # angle subtended by the two goal posts at shot location (radians)
    vx1 = (LEFT_POST[0] - x, LEFT_POST[1] - y)
    vx2 = (RIGHT_POST[0] - x, RIGHT_POST[1] - y)
    # angle between vectors
    dot = vx1[0]*vx2[0] + vx1[1]*vx2[1]
    mag1 = math.hypot(vx1[0], vx1[1])
    mag2 = math.hypot(vx2[0], vx2[1])
    if mag1 * mag2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(cosang)  # radians

def parse_statsbomb_events_file(path: Path):
    events = load_events_json(path)
    shots = []
    for ev in events:
        if ev.get("type", {}).get("name") == "Shot":
            shot = {}
            shot["match_id"] = ev.get("match_id")
            shot["event_id"] = ev.get("id")
            shot["player_id"] = ev.get("player", {}).get("id")
            shot["player_name"] = ev.get("player", {}).get("name")
            shot["team_id"] = ev.get("team", {}).get("id")
            shot["team_name"] = ev.get("team", {}).get("name")
            shot["minute"] = ev.get("minute")
            shot["second"] = ev.get("second")
            # outcome/subtype
            shot["shot_outcome"] = ev.get("shot", {}).get("outcome", {}).get("name")
            shot["shot_body_part"] = ev.get("shot", {}).get("body_part", {}).get("name")
            shot["shot_technique"] = ev.get("shot", {}).get("technique", {}).get("name")
            # location
            loc = ev.get("location", [None, None])
            shot["x"] = float(loc[0]) if loc[0] is not None else None
            shot["y"] = float(loc[1]) if loc[1] is not None else None
            if shot["x"] is not None and shot["y"] is not None:
                shot["dist_to_goal"] = shot_distance(shot["x"], shot["y"])
                shot["angle_to_goal_rad"] = shot_angle(shot["x"], shot["y"])
                shot["angle_to_goal_deg"] = math.degrees(shot["angle_to_goal_rad"])
            else:
                shot["dist_to_goal"] = None
                shot["angle_to_goal_rad"] = None
                shot["angle_to_goal_deg"] = None

            shots.append(shot)
    return pd.DataFrame(shots)

def collect_all_shots(events_dir: Path):
    dfs = []
    for path in events_dir.glob("**/*.json"):
        try:
            df = parse_statsbomb_events_file(path)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"failed to parse {path}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()
    

def plot_shots(df, max_shots=None, figsize=(10, 6), show_outcome=True):
    """
    Plot football shots on a StatsBomb pitch (120x80).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing shot data with columns: x, y, shot_outcome
    max_shots : int, optional
        Maximum number of shots to display (None = all shots)
    figsize : tuple, optional
        Figure size (width, height)
    show_outcome : bool, optional
        Color code shots by outcome
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect('equal')
    
    # Draw pitch elements
    # Pitch outline
    ax.plot([0, 120, 120, 0, 0], [0, 0, 80, 80, 0], color='white', linewidth=2)
    
    # Center line
    ax.plot([60, 60], [0, 80], color='white', linewidth=2)
    
    # Center circle
    center_circle = plt.Circle((60, 40), 9.15, fill=False, color='white', linewidth=2)
    ax.add_patch(center_circle)
    
    # Center spot
    ax.scatter(60, 40, color='white', s=20, zorder=3)
    
    # Left penalty area
    ax.plot([0, 18, 18, 0], [18, 18, 62, 62], color='white', linewidth=2)
    
    # Left 6-yard box
    ax.plot([0, 6, 6, 0], [30, 30, 50, 50], color='white', linewidth=2)
    
    # Right penalty area
    ax.plot([120, 102, 102, 120], [18, 18, 62, 62], color='white', linewidth=2)
    
    # Right 6-yard box
    ax.plot([120, 114, 114, 120], [30, 30, 50, 50], color='white', linewidth=2)
    
    # Left penalty arc
    left_arc = Arc((12, 40), 18.3, 18.3, angle=0, theta1=310, theta2=50, color='white', linewidth=2)
    ax.add_patch(left_arc)
    
    # Right penalty arc
    right_arc = Arc((108, 40), 18.3, 18.3, angle=0, theta1=130, theta2=230, color='white', linewidth=2)
    ax.add_patch(right_arc)
    
    # Penalty spots
    ax.scatter([12, 108], [40, 40], color='white', s=20, zorder=3)
    
    # Goals
    ax.plot([0, 0], [36, 44], color='white', linewidth=4)
    ax.plot([120, 120], [36, 44], color='white', linewidth=4)
    
    # Define colors for shot outcomes
    outcome_colors = {
        'Goal': '#00FF00',
        'Saved': '#FFA500',
        'Off T': '#FF0000',
        'Blocked': '#FFFF00',
        'Wayward': '#FF69B4',
        'Post': '#00FFFF',
        'Saved Off Target': '#FF8C00',
        'Saved to Post': '#87CEEB'
    }
    
    # Limit shots if specified
    plot_df = df.head(max_shots) if max_shots else df
    
    # Plot shots
    for idx, shot in plot_df.iterrows():
        if show_outcome and 'shot_outcome' in shot:
            color = outcome_colors.get(shot['shot_outcome'], '#FFFFFF')
            label = shot['shot_outcome']
        else:
            color = '#FFFFFF'
            label = None
        
        # Plot shot location
        ax.scatter(shot['x'], shot['y'], 
                  c=color, s=100, alpha=0.7, 
                  edgecolors='black', linewidth=1.5,
                  zorder=4)
        
        # Draw line to goal
        ax.plot([shot['x'], 120], [shot['y'], 40], 
               color=color, alpha=0.3, linewidth=1, linestyle='--')
    
    # Create legend
    if show_outcome and 'shot_outcome' in df.columns:
        unique_outcomes = plot_df['shot_outcome'].unique()
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=outcome_colors.get(outcome, '#FFFFFF'), 
                                     markersize=10, label=outcome, 
                                     markeredgecolor='black', markeredgewidth=1.5)
                          for outcome in unique_outcomes if outcome in outcome_colors]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.8)
    
    # Styling
    ax.set_facecolor('#195905')
    fig.patch.set_facecolor('#195905')
    ax.axis('off')
    
    plt.title('Shot Map', color='white', size=16, weight='bold', pad=20)
    plt.tight_layout()
    
    return fig, ax