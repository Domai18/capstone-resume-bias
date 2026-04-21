"""
Visualization Code for Capstone Project: Algorithmic Bias in Resume Screening
Author: Derrick Omai
Course: CDA 490 Capstone, Spring 2026

This file contains all visualization code used in the project.
Each function generates a specific figure used in the analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Colorblind-friendly palette (Wong's palette)
COLOR_WHITE = '#0072B2'      # Blue - for White applicants
COLOR_BLACK = '#D55E00'      # Vermillion/Orange - for Black applicants
COLOR_LR = '#56B4E9'         # Sky blue - for Logistic Regression
COLOR_RF = '#E69F00'         # Orange/Amber - for Random Forest

# Output directory for figures
FIGURES_DIR = Path('results/figures')


# ============================================================================
# FIGURE 1: FEATURES BY FIELD
# Purpose: Show average skills, experience, and education across job fields
# ============================================================================

def plot_features_by_field(df):
    """
    Creates a 3-panel bar chart showing average features by professional field.

    Parameters:
        df: DataFrame with columns ['Field', 'skills_count', 'years_experience', 'education_level']
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    features = ['skills_count', 'years_experience', 'education_level']
    titles = ['Skills Count', 'Years of Experience', 'Education Level']

    for ax, feature, title in zip(axes, features, titles):
        df.groupby('Field')[feature].mean().plot(
            kind='bar', ax=ax, color='steelblue', edgecolor='black'
        )
        ax.set_title(f'Average {title} by Field')
        ax.set_xlabel('')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'features_by_field.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# FIGURE 2: RACIAL DISTRIBUTION
# Purpose: Verify balanced racial groups in matched-pair design
# ============================================================================

def plot_racial_distribution(df):
    """
    Creates a 2-panel visualization showing racial distribution.
    Left: Overall counts by race
    Right: Counts by race within each professional field

    Parameters:
        df: DataFrame with columns ['race', 'Field']
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['steelblue', 'coral']

    # Panel 1: Records by race
    race_counts = df['race'].value_counts()
    race_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
    axes[0].set_title('Records by Race')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['Black', 'White'], rotation=0)
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')

    # Panel 2: Records by race and field
    race_field = df.groupby(['Field', 'race']).size().unstack()
    race_field.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
    axes[1].set_title('Records by Field and Race')
    axes[1].set_xlabel('')
    axes[1].legend(title='Race')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'racial_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# FIGURE 3: QUALIFICATION SCORE DISTRIBUTION
# Purpose: Prove matched pairs have identical qualifications (box plots)
# ============================================================================

def plot_qualification_scores(df):
    """
    Creates a 2-panel visualization:
    Left: Overall qualification score histogram with median line
    Right: Box plots comparing White vs Black (should be identical)

    Parameters:
        df: DataFrame with columns ['race', 'qualification_score'] or
            ['skills_count', 'years_experience', 'education_level']
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Compute qualification score if not present
    if 'qualification_score' not in df.columns:
        df['qualification_score'] = (
            0.35 * df['skills_count'] +
            0.40 * df['years_experience'] +
            0.25 * df['education_level']
        )

    score_col = 'qualification_score'
    white_scores = df[df['race'] == 'white'][score_col]
    black_scores = df[df['race'] == 'black'][score_col]

    # LEFT PANEL: Overall distribution histogram
    axes[0].hist(df[score_col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    overall_median = df[score_col].median()
    axes[0].axvline(x=overall_median, color='red', linestyle='--', linewidth=2.5,
                    label=f'Median: {overall_median:.1f}')
    axes[0].set_title('Overall Qualification Score Distribution', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Qualification Score', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    # RIGHT PANEL: Box plots by race
    box_data = [white_scores, black_scores]
    bp = axes[1].boxplot(box_data, tick_labels=['White\n(n=551)', 'Black\n(n=551)'],
                          patch_artist=True, widths=0.5)

    # Style the boxes
    colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    # Add median value annotations
    axes[1].annotate(f'Median: {white_scores.median():.1f}',
                     xy=(1, white_scores.median()),
                     xytext=(0.55, white_scores.median() + 2),
                     fontsize=10, fontweight='bold')
    axes[1].annotate(f'Median: {black_scores.median():.1f}',
                     xy=(2, black_scores.median()),
                     xytext=(2.1, black_scores.median() + 2),
                     fontsize=10, fontweight='bold')

    axes[1].set_title('Qualification Score by Race\n(Identical by Matched-Pair Design)',
                      fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Qualification Score', fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qualification_scores.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# FIGURE 4: CALLBACK RATE BY RACE (Logistic Regression)
# Purpose: Show predicted callback rates by race across experimental regimes
# ============================================================================

def plot_callback_rate_lr(df):
    """
    Creates a bar chart showing callback rates by race for Logistic Regression.

    Parameters:
        df: DataFrame with prediction probability columns for each regime
    """
    regime_names = ['A: Quals', 'B: +Names', 'C: +Race']
    regimes = ['A_quals_only', 'B_quals_plus_names', 'C_quals_plus_race']

    white_rates = []
    black_rates = []

    for regime in regimes:
        prob_col = f'{regime}_Logistic_Regression_prob'
        white_rates.append(df[df['race'] == 'white'][prob_col].mean())
        black_rates.append(df[df['race'] == 'black'][prob_col].mean())

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(regime_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, white_rates, width, label='White',
                   color='steelblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, black_rates, width, label='Black',
                   color='coral', edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bar, rate in zip(bars1, white_rates):
        ax.annotate(f'{rate:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')
    for bar, rate in zip(bars2, black_rates):
        ax.annotate(f'{rate:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')

    ax.set_ylabel('Predicted Callback Probability', fontsize=12)
    ax.set_xlabel('Experimental Regime', fontsize=12)
    ax.set_title('Callback Rate by Race (Logistic Regression)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regime_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, max(white_rates + black_rates) * 1.25)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'callback_rate_by_race_lr.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# FIGURE 5: CALLBACK RATE BY RACE (Random Forest)
# Purpose: Show predicted callback rates by race for RF model
# ============================================================================

def plot_callback_rate_rf(df):
    """
    Creates a bar chart showing callback rates by race for Random Forest.

    Parameters:
        df: DataFrame with prediction probability columns for each regime
    """
    regime_names = ['A: Quals', 'B: +Names', 'C: +Race']
    regimes = ['A_quals_only', 'B_quals_plus_names', 'C_quals_plus_race']

    white_rates = []
    black_rates = []

    for regime in regimes:
        prob_col = f'{regime}_Random_Forest_prob'
        white_rates.append(df[df['race'] == 'white'][prob_col].mean())
        black_rates.append(df[df['race'] == 'black'][prob_col].mean())

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(regime_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, white_rates, width, label='White',
                   color='steelblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, black_rates, width, label='Black',
                   color='coral', edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bar, rate in zip(bars1, white_rates):
        ax.annotate(f'{rate:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')
    for bar, rate in zip(bars2, black_rates):
        ax.annotate(f'{rate:.1%}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center',
                    fontsize=10, fontweight='bold')

    ax.set_ylabel('Predicted Callback Probability', fontsize=12)
    ax.set_xlabel('Experimental Regime', fontsize=12)
    ax.set_title('Callback Rate by Race (Random Forest)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(regime_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, max(white_rates + black_rates) * 1.25)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'callback_rate_by_race_rf.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# FIGURE 6: CALLBACK RATIO AND DISPARATE IMPACT (Colorblind-Friendly)
# Purpose: Compare LR vs RF across regimes with legal threshold lines
# ============================================================================

def plot_callback_ratio_and_disparate_impact(df):
    """
    Creates a 2-panel visualization comparing both models:
    Left: White:Black callback ratio by regime
    Right: Disparate impact ratio with 80% rule threshold

    Uses colorblind-friendly palette with hatching for model differentiation.

    Parameters:
        df: DataFrame with prediction probability columns for each regime and model
    """
    regime_names = ['A: Quals Only', 'B: +Names', 'C: +Race']
    regimes = ['A_quals_only', 'B_quals_plus_names', 'C_quals_plus_race']

    # Calculate callback probabilities for both models
    lr_white, lr_black, rf_white, rf_black = [], [], [], []

    for regime in regimes:
        lr_white.append(df[df['race'] == 'white'][f'{regime}_Logistic_Regression_prob'].mean())
        lr_black.append(df[df['race'] == 'black'][f'{regime}_Logistic_Regression_prob'].mean())
        rf_white.append(df[df['race'] == 'white'][f'{regime}_Random_Forest_prob'].mean())
        rf_black.append(df[df['race'] == 'black'][f'{regime}_Random_Forest_prob'].mean())

    # Calculate ratios
    lr_ratio = [w/b if b > 0 else 0 for w, b in zip(lr_white, lr_black)]
    rf_ratio = [w/b if b > 0 else 0 for w, b in zip(rf_white, rf_black)]

    # Calculate disparate impact (Black/White)
    lr_di = [b/w if w > 0 else 0 for w, b in zip(lr_white, lr_black)]
    rf_di = [b/w if w > 0 else 0 for w, b in zip(rf_white, rf_black)]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(regime_names))
    width = 0.35

    # ---- Panel 1: Predicted Callback Ratio ----
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, lr_ratio, width, label='Logistic Regression',
                    color=COLOR_LR, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, rf_ratio, width, label='Random Forest',
                    color=COLOR_RF, edgecolor='black', linewidth=1.2, hatch='///')

    # Add value labels
    for bar, val in zip(bars1, lr_ratio):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center',
                     fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, rf_ratio):
        ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center',
                     fontsize=10, fontweight='bold')

    # Reference lines
    ax1.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Parity (1.0)')
    ax1.axhline(y=1.36, color='red', linestyle=':', linewidth=2, label='Training Labels (1.36)')

    ax1.set_ylabel('White:Black Callback Ratio', fontsize=12)
    ax1.set_xlabel('Experimental Regime', fontsize=12)
    ax1.set_title('Predicted Callback Ratio by Regime', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_names, fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_ylim(0, max(lr_ratio + rf_ratio) * 1.2)

    # ---- Panel 2: Disparate Impact ----
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, lr_di, width, label='Logistic Regression',
                    color=COLOR_LR, edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x + width/2, rf_di, width, label='Random Forest',
                    color=COLOR_RF, edgecolor='black', linewidth=1.2, hatch='///')

    # Add value labels
    for bar, val in zip(bars3, lr_di):
        ax2.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center',
                     fontsize=10, fontweight='bold')
    for bar, val in zip(bars4, rf_di):
        ax2.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points', ha='center',
                     fontsize=10, fontweight='bold')

    # Reference lines
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Parity (1.0)')
    ax2.axhline(y=0.8, color='red', linestyle=':', linewidth=2, label='80% Rule Threshold')

    ax2.set_ylabel('Disparate Impact Ratio (Black/White)', fontsize=12)
    ax2.set_xlabel('Experimental Regime', fontsize=12)
    ax2.set_title('Disparate Impact by Regime', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regime_names, fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(0, 1.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'callback_ratio_and_disparate_impact.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN: Example usage
# ============================================================================

if __name__ == "__main__":
    # Load data
    eda_data = pd.read_csv('data/processed/resumes_with_features.csv')
    predictions_data = pd.read_csv('data/processed/test_set_predictions.csv')

    print("Generating visualizations...\n")

    # Generate each figure
    print("1. Features by Field")
    plot_features_by_field(eda_data)

    print("2. Racial Distribution")
    plot_racial_distribution(eda_data)

    print("3. Qualification Scores")
    plot_qualification_scores(eda_data)

    print("4. Callback Rate (Logistic Regression)")
    plot_callback_rate_lr(predictions_data)

    print("5. Callback Rate (Random Forest)")
    plot_callback_rate_rf(predictions_data)

    print("6. Callback Ratio and Disparate Impact")
    plot_callback_ratio_and_disparate_impact(predictions_data)

    print("\nAll figures saved to:", FIGURES_DIR)
