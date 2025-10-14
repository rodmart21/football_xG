import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def create_attacking_indices(df):
    """Create comprehensive attacking performance indices"""
    
    # Offensive Contribution Index (weighted combination)
    df['Offensive_Contribution_Index'] = (
        df['Goals_per90'] * 0.30 +
        df['xG_per90'] * 0.15 +
        df['Shot_Creating_Action_per90'] * 0.20 +
        df['Progressive_Carries_per_90'] * 0.15 +
        df['Take_Ons_Succ_per_90'] * 0.20
    )
    
    # Creativity Score
    df['Creativity_Score'] = (
        df['Key_Passes_per_90'] +
        df['Progressive_Passes_per_90'] +
        df['Passes_1/3_per_90']
    )
    
    # Space Generation (ability to draw fouls and beat defenders)
    df['Space_Generation'] = (
        df['Take_Ons_Succ_per_90'] * df['Fouls_Drawn_per_90']
    )
    
    # Efficiency Score (output per touch)
    df['Efficiency_Score'] = (
        (df['Goals_per90'] + df['Assists_per_90']) / 
        (df['Touches_per_90'] / 10)  # normalize
    )
    
    # Penetration Score (ability to get into dangerous areas)
    df['Penetration_Score'] = (
        df['Touches_Att_Pen_per_90'] +
        df['Carries_Penalty_Area_per_90'] +
        df['Progressive_Carries_per_90']
    )
    
    # Finishing Quality
    df['Finishing_Quality'] = df['Goals_per90'] - df['xG_per90']
    
    # Shot Quality
    df['Shot_Quality'] = df['Shots_on_target_per90'] / (df['Shots_total_per90'] + 0.01)
    
    return df

def create_radar_chart(player_row, df):
    """Create radar chart for an individual player (Series)"""
    categories = ['Goals', 'Assists', 'Shot Creation', 
                  'Dribbling', 'Progression', 'Penetration']
    
    # Use league percentiles from the whole DataFrame
    values = [
        (player_row['Goals_per90'] / df['Goals_per90'].quantile(0.95)) * 100,
        (player_row['Assists_per_90'] / df['Assists_per_90'].quantile(0.95)) * 100,
        (player_row['Shot_Creating_Action_per90'] / df['Shot_Creating_Action_per90'].quantile(0.95)) * 100,
        (player_row['Take_Ons_Succ_per_90'] / df['Take_Ons_Succ_per_90'].quantile(0.95)) * 100,
        (player_row['Progressive_Carries_per_90'] / df['Progressive_Carries_per_90'].quantile(0.95)) * 100,
        # (player_row['Creativity_Score'] / df_filtered['Creativity_Score'].quantile(0.95)) * 100
        (player_row['Penetration_Score'] / df['Penetration_Score'].quantile(0.95)) * 100
    ]
    
    # Cap at 100
    values = [min(v, 100) for v in values]
    
    # Number of variables
    N = len(categories)
    
    # Compute angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#1f77b4')
    ax.fill(angles, values, alpha=0.25, color='#1f77b4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 100)
    
    ax.set_title(
        f"{player_row['Player']} - Complete Attacker Profile\n(Percentile Rankings)",
        size=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(True)
    
    return fig


def comprehensive_data_quality_check(df):
    """
    Comprehensive data quality check for football statistics.
    Identifies outliers, missing values, inconsistencies, and data issues.
    
    Parameters:
    df (pd.DataFrame): Your football data
    
    Returns:
    dict: Dictionary with all detected issues and statistics
    """
    
    print("="*80)
    print("FOOTBALL DATA QUALITY REPORT")
    print("="*80)
    
    issues = {
        'missing_data': {},
        'outliers': {},
        'logical_errors': {},
        'suspicious_values': {},
        'data_type_issues': {}
    }
    
    # ==========================================
    # 1. BASIC DATA OVERVIEW
    # ==========================================
    print(f"\n{'='*80}")
    print("1. BASIC OVERVIEW")
    print(f"{'='*80}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Duplicated rows: {df.duplicated().sum()}")
    
    # ==========================================
    # 2. MISSING DATA ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print("2. MISSING DATA ANALYSIS")
    print(f"{'='*80}")
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n⚠️  Found {len(missing_df)} columns with missing data:")
        print(missing_df.head(20))
        issues['missing_data'] = missing_df.to_dict()
    else:
        print("✅ No missing data found!")
    
    # ==========================================
    # 3. OUTLIER DETECTION (Z-Score Method)
    # ==========================================
    print(f"\n{'='*80}")
    print("3. OUTLIER DETECTION (|Z-Score| > 3)")
    print(f"{'='*80}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['Unnamed: 0', 'Age', 'Born']]
    
    outlier_summary = {}
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = np.sum(z_scores > 3)
            if outliers > 0:
                outlier_summary[col] = outliers
    
    if outlier_summary:
        print(f"\n⚠️  Found outliers in {len(outlier_summary)} columns:")
        outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier_Count'])
        outlier_df = outlier_df.sort_values('Outlier_Count', ascending=False)
        print(outlier_df.head(20))
        issues['outliers'] = outlier_summary
    else:
        print("✅ No extreme outliers detected!")
    
    # ==========================================
    # 4. LOGICAL CONSISTENCY CHECKS
    # ==========================================
    print(f"\n{'='*80}")
    print("4. LOGICAL CONSISTENCY CHECKS")
    print(f"{'='*80}")
    
    logical_issues = []
    
    # Check 1: Minutes played should be >= 0
    if 'Min' in df.columns:
        negative_minutes = (df['Min'] < 0).sum()
        if negative_minutes > 0:
            logical_issues.append(f"⚠️  {negative_minutes} players with negative minutes")
    
    # Check 2: Age should be reasonable (15-45)
    if 'Age' in df.columns:
        unreasonable_age = ((df['Age'] < 15) | (df['Age'] > 45)).sum()
        if unreasonable_age > 0:
            logical_issues.append(f"⚠️  {unreasonable_age} players with unreasonable age (<15 or >45)")
    
    # Check 3: Percentages should be 0-100
    percentage_cols = [col for col in df.columns if '%' in col or 'Percentage' in col]
    for col in percentage_cols:
        if col in df.columns:
            invalid_pct = ((df[col] < 0) | (df[col] > 100)).sum()
            if invalid_pct > 0:
                logical_issues.append(f"⚠️  {invalid_pct} invalid values in {col} (should be 0-100)")
    
    # Check 4: Starts should be <= Matches Played
    if 'MP' in df.columns and 'Starts' in df.columns:
        invalid_starts = (df['Starts'] > df['MP']).sum()
        if invalid_starts > 0:
            logical_issues.append(f"⚠️  {invalid_starts} players with more starts than matches played")
    
    # Check 5: Goals should be <= Shots
    if 'Shots_total_per90' in df.columns and 'Goals_per90' in df.columns:
        # This is per 90, so it's possible but unlikely
        invalid_goals = (df['Goals_per90'] > df['Shots_total_per90']).sum()
        if invalid_goals > 0:
            logical_issues.append(f"⚠️  {invalid_goals} players with more goals than shots per 90 (unusual)")
    
    # Check 6: Assists should match pass completion
    if 'Passes_Total_Cmp' in df.columns:
        zero_passes = (df['Passes_Total_Cmp'] == 0).sum()
        if zero_passes > 0:
            logical_issues.append(f"⚠️  {zero_passes} players with zero completed passes (check if valid)")
    
    # Check 7: xG should be somewhat correlated with shots
    if 'xG_per90' in df.columns and 'Shots_total_per90' in df.columns:
        high_xg_no_shots = ((df['xG_per90'] > 0.5) & (df['Shots_total_per90'] < 0.5)).sum()
        if high_xg_no_shots > 0:
            logical_issues.append(f"⚠️  {high_xg_no_shots} players with high xG but very few shots (data mismatch?)")
    
    if logical_issues:
        for issue in logical_issues:
            print(issue)
        issues['logical_errors'] = logical_issues
    else:
        print("✅ No logical inconsistencies found!")
    
    # ==========================================
    # 5. SUSPICIOUS VALUES
    # ==========================================
    print(f"\n{'='*80}")
    print("5. SUSPICIOUS VALUES (Top 5 Extreme Values)")
    print(f"{'='*80}")
    
    key_metrics = ['Goals_per90', 'Assists_per_90', 'Shots_total_per90', 
                   'Touches_per_90', 'Take_Ons_Succ%', 'Yellow_Cards_per_90']
    
    for col in key_metrics:
        if col in df.columns:
            top_values = df.nlargest(5, col)[['Player', col]]
            print(f"\nTop 5 {col}:")
            print(top_values.to_string(index=False))
            
            # Flag extremely high values
            if col == 'Goals_per90' and df[col].max() > 2.0:
                issues['suspicious_values'][col] = f"Extremely high: {df[col].max():.2f}"
            elif col == 'Touches_per_90' and df[col].max() > 150:
                issues['suspicious_values'][col] = f"Extremely high: {df[col].max():.2f}"
    
    # ==========================================
    # 6. DATA TYPE ISSUES
    # ==========================================
    print(f"\n{'='*80}")
    print("6. DATA TYPE CHECK")
    print(f"{'='*80}")
    
    # Check if numeric columns are actually numeric
    for col in df.columns:
        if 'per90' in col or 'per_90' in col or col.startswith('Passes_') or col.startswith('Shots_'):
            if df[col].dtype == 'object':
                print(f"⚠️  '{col}' should be numeric but is type: {df[col].dtype}")
                issues['data_type_issues'][col] = str(df[col].dtype)
    
    if not issues['data_type_issues']:
        print("✅ All expected numeric columns have correct data types!")
    
    # ==========================================
    # 7. DISTRIBUTION ANALYSIS (Visual)
    # ==========================================
    print(f"\n{'='*80}")
    print("7. GENERATING DISTRIBUTION PLOTS...")
    print(f"{'='*80}")
    
    key_metrics_plot = ['Goals_per90', 'Assists_per_90', 'xG_per90', 
                        'Shots_total_per90', 'Passes_Total_Cmp%', 'Take_Ons_Succ%']
    key_metrics_plot = [col for col in key_metrics_plot if col in df.columns]
    
    if key_metrics_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Distribution Analysis - Key Metrics', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(key_metrics_plot):
            row = idx // 3
            col_idx = idx % 3
            
            # Remove outliers for better visualization
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            filtered_data = df[col][(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            axes[row, col_idx].hist(filtered_data, bins=50, edgecolor='black', alpha=0.7)
            axes[row, col_idx].axvline(df[col].mean(), color='red', linestyle='--', 
                                       linewidth=2, label=f'Mean: {df[col].mean():.2f}')
            axes[row, col_idx].axvline(df[col].median(), color='green', linestyle='--', 
                                       linewidth=2, label=f'Median: {df[col].median():.2f}')
            axes[row, col_idx].set_title(col, fontweight='bold')
            axes[row, col_idx].legend(fontsize=8)
            axes[row, col_idx].set_xlabel('Value')
            axes[row, col_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('data_quality_distributions.png', dpi=300, bbox_inches='tight')
        print("✅ Distribution plots saved as 'data_quality_distributions.png'")
        plt.show()
    
    # ==========================================
    # 8. CORRELATION SANITY CHECKS
    # ==========================================
    print(f"\n{'='*80}")
    print("8. CORRELATION SANITY CHECKS")
    print(f"{'='*80}")
    
    # Check if related metrics are correlated as expected
    correlation_checks = [
        ('Goals_per90', 'xG_per90', 0.5),  # Should be positively correlated
        ('Shots_total_per90', 'Goals_per90', 0.3),  # Should be positively correlated
        ('Passes_Total_Cmp', 'Passes_Total_Att', 0.95),  # Should be highly correlated
        ('Assists_per_90', 'Key_Passes_per_90', 0.3),  # Should be positively correlated
    ]
    
    for col1, col2, expected_min_corr in correlation_checks:
        if col1 in df.columns and col2 in df.columns:
            corr = df[[col1, col2]].corr().iloc[0, 1]
            if abs(corr) < expected_min_corr:
                print(f"⚠️  Weak correlation between {col1} and {col2}: {corr:.3f} (expected > {expected_min_corr})")
            else:
                print(f"✅ {col1} vs {col2}: {corr:.3f}")
    
    # ==========================================
    # 9. SUMMARY REPORT
    # ==========================================
    print(f"\n{'='*80}")
    print("9. SUMMARY REPORT")
    print(f"{'='*80}")
    
    total_issues = (
        len(issues['missing_data']) +
        len(issues['outliers']) +
        len(issues['logical_errors']) +
        len(issues['suspicious_values']) +
        len(issues['data_type_issues'])
    )
    
    print(f"\nTotal Issues Found: {total_issues}")
    print(f"  - Missing Data: {len(issues['missing_data'])} columns")
    print(f"  - Outliers: {len(issues['outliers'])} columns")
    print(f"  - Logical Errors: {len(issues['logical_errors'])} issues")
    print(f"  - Suspicious Values: {len(issues['suspicious_values'])} columns")
    print(f"  - Data Type Issues: {len(issues['data_type_issues'])} columns")
    
    if total_issues == 0:
        print("\n🎉 Your data looks clean and ready for analysis!")
    else:
        print("\n⚠️  Please review the issues above before proceeding with analysis.")
    
    # ==========================================
    # 10. RECOMMENDATIONS
    # ==========================================
    print(f"\n{'='*80}")
    print("10. RECOMMENDATIONS")
    print(f"{'='*80}")
    
    recommendations = []
    
    if len(issues['missing_data']) > 0:
        recommendations.append("• Handle missing data: Consider imputation or filtering rows/columns")
    
    if len(issues['outliers']) > 10:
        recommendations.append("• Review outliers: Verify if they're legitimate or data errors")
    
    if issues['logical_errors']:
        recommendations.append("• Fix logical inconsistencies before analysis")
    
    if issues['data_type_issues']:
        recommendations.append("• Convert columns to proper numeric types")
    
    recommendations.append("• Consider filtering players with minimum minutes played (e.g., 900+ minutes)")
    recommendations.append("• Normalize per-90 metrics for fair comparison")
    
    for rec in recommendations:
        print(rec)
    
    print(f"\n{'='*80}")
    print("DATA QUALITY CHECK COMPLETE!")
    print(f"{'='*80}\n")
    
    return issues


