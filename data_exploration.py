import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FeatureExplorer:
    def __init__(self, features_csv: str, metadata_csv: str):
        """
        Initialize the feature explorer with paths to the feature CSV and metadata CSV
        containing the target values (Brix scores).
        """
        self.features_df = pd.read_csv(features_csv)
        self.metadata_df = pd.read_csv(metadata_csv)
        self.target_values = None
        self.feature_importances = None
        self.pca = None
        self.pca_components = None
        self.shap_values = None
        
    def prepare_data(self):
        """Prepare and merge the data, computing mean features per watermelon"""
        # Group by watermelon_id and compute mean of all numeric features
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        watermelon_features = self.features_df.groupby('watermelon_id')[numeric_cols].mean()
        
        # Merge with metadata to get target values
        merged_data = watermelon_features.merge(
            self.metadata_df, 
            on='watermelon_id'
        )

        merged_data = merged_data.dropna()
        
        self.target_values = merged_data['brix_score']
        self.features = merged_data.drop(['brix_score', 'watermelon_id'], axis=1)
        
        return merged_data
    
    def analyze_feature_importance(self):
        """Calculate feature importance using multiple metrics including neural network SHAP values"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.features)
        
        # 1. Calculate mutual information scores
        mi_scores = mutual_info_regression(X_scaled, self.target_values)

        # 2. Calculate F-scores
        f_scores, _ = f_regression(X_scaled, self.target_values)
        
        # 3. Calculate Spearman correlations - keep actual values
        spearman_corrs = np.array([
            spearmanr(self.features[col], self.target_values)[0] 
            for col in self.features.columns
        ])
        
        # 4. Calculate PCA components
        self.pca = PCA()
        self.pca_components = self.pca.fit_transform(X_scaled)
        pca_importance = np.abs(self.pca.components_[0])
        
        # Neural Network and SHAP values calculation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, 
            self.target_values, 
            test_size=0.2, 
            random_state=42
        )
        
        # Create data loaders
        train_dataset = WatermelonDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        
        # Initialize and train model
        model = BrixPredictor(input_dim=X_scaled.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        
        # Training loop
        n_epochs = 150
        model.train()
        for epoch in range(n_epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Calculate SHAP values using DeepExplainer
        model.eval()
        background = torch.FloatTensor(X_train[:5000])  # Use first 1000 training samples as background
        test_tensor = torch.FloatTensor(X_scaled)
        
        explainer = shap.DeepExplainer(model, background)
        self.shap_values = explainer.shap_values(test_tensor)
        
        # If shap_values is a list (common with DeepExplainer), take first element
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]
        
        shap_importance = np.abs(self.shap_values).mean(0).squeeze()  # Use squeeze to remove extra dimension
        print(f"SHAP importance (shape: {shap_importance.shape}): {shap_importance}")
        
        # Create importance DataFrame with all metrics
        self.feature_importances = pd.DataFrame({
            'feature': self.features.columns,
            'mutual_info': mi_scores,
            'f_score': f_scores,
            'spearman_corr': spearman_corrs,
            'pca_importance': pca_importance,
            'shap_importance': shap_importance
        })
        
        # Add rankings for each metric
        for metric in ['mutual_info', 'f_score', 'pca_importance', 'shap_importance']:
            self.feature_importances[f'{metric}_rank'] = self.feature_importances[metric].rank(ascending=False)
        
        self.feature_importances['spearman_corr_rank'] = self.feature_importances['spearman_corr'].abs().rank(ascending=False)
        
        return self.feature_importances

    def plot_feature_analysis(self, n_features=20, output_dir='figures'):
        """Plot comprehensive feature analysis and save to separate files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Statistical Metrics Figure (MI, F-Score, Spearman)
        fig_stats, axes_stats = plt.subplots(1, 3, figsize=(20, 10))
        
        # Plot MI
        data_mi = self.feature_importances.sort_values('mutual_info', ascending=True).tail(n_features)
        sns.barplot(
            data=data_mi,
            x='mutual_info',
            y='feature',
            ax=axes_stats[0],
            palette='viridis'
        )
        axes_stats[0].set_title('Mutual Information\nHigher = stronger relationship')
        
        # Plot F-Score
        data_f = self.feature_importances.sort_values('f_score', ascending=True).tail(n_features)
        sns.barplot(
            data=data_f,
            x='f_score',
            y='feature',
            ax=axes_stats[1],
            palette='viridis'
        )
        axes_stats[1].set_title('F-Score\nHigher = stronger linear relationship')
        
        # Sort features by SHAP importance
        shap_importance = pd.DataFrame({
            'feature': self.features.columns,
            'importance': np.abs(self.shap_values).mean(0).squeeze()
        }).sort_values('importance', ascending=True).tail(n_features)
        
        # Plot SHAP importance values
        sns.barplot(
            data=shap_importance,
            x='importance',
            y='feature',
            palette='viridis'
        )
        
        axes_stats[2].set_title('SHAP Feature Importance\nAverage impact on model predictions')
        
        plt.tight_layout()
        fig_stats.savefig(f'{output_dir}/stats_and_shap.png', dpi=300, bbox_inches='tight')
        plt.close(fig_stats)

        # 2. PCA Components Figure
        n_components = min(3, len(self.features.columns))   
        fig_pca, axes_pca = plt.subplots(1, n_components, figsize=(20, 10))
        
        for i in range(n_components):
            pca_importance = np.abs(self.pca.components_[i])
            pca_df = pd.DataFrame({
                'feature': self.features.columns,
                'importance': pca_importance
            }).sort_values('importance', ascending=True).tail(n_features)
            
            sns.barplot(
                data=pca_df,
                x='importance',
                y='feature',
                ax=axes_pca[i],
                palette='viridis'
            )
            axes_pca[i].set_title(f'PCA Component {i+1}\nVariance Explained: {self.pca.explained_variance_ratio_[i]:.3f}')
        
        plt.tight_layout()
        fig_pca.savefig(f'{output_dir}/pca_components.png', dpi=300, bbox_inches='tight')
        plt.close(fig_pca)

        # Get top features from Spearman correlation to include significant features from this metric in the correlation heatmap
        data_sp = self.feature_importances.nlargest(n_features, 'spearman_corr_rank')

        # 3. Feature Correlation Heatmap
        fig_corr = plt.figure(figsize=(20, 10))
        
        # Get top features from statistical metrics
        top_features = pd.unique(pd.concat([
            data_mi['feature'],
            data_f['feature'],
            data_sp['feature']
        ]))
        
        correlation_data = pd.concat([
            self.features[top_features],
            self.target_values.rename('brix_score')
        ], axis=1)
        
        sns.heatmap(
            correlation_data.corr(),
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Feature Correlation Heatmap\nShows relationships between features and target')
        plt.tight_layout()
        fig_corr.savefig(f'{output_dir}/feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close(fig_corr)

        # 4. Spearman Correlation Heatmap
        fig_spearman = plt.figure(figsize=(20, 10))
        top_spearman_features = data_sp['feature'].tolist()
        spearman_matrix = self.features[top_spearman_features].corr(method='spearman')
        # Create heatmap
        sns.heatmap(
            spearman_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            cbar_kws={'label': 'Spearman Correlation Coefficient'}
        )
        plt.title('Spearman Correlation Heatmap\nFeature-to-Feature Correlations')
        plt.tight_layout()
        fig_spearman.savefig(f'{output_dir}/spearman_correlation.png', dpi=300, bbox_inches='tight')
        plt.close(fig_spearman)

        # 5. SHAP Summary Plot 
        plt.figure(figsize=(20, 10))
        print(f"SHAP values shape before summary plot: {self.shap_values.shape}")
        print(f"Features shape: {self.features.shape}")

        # Try explicitly reshaping the SHAP values if needed
        if len(self.shap_values.shape) > 2:
            self.shap_values = self.shap_values.reshape(self.features.shape[0], -1)
            print(f"Reshaped SHAP values: {self.shap_values.shape}")

        # Create a new DataFrame to ensure proper alignment
        feature_df = self.features.copy()

        # Use plot_type="dot" to force display of all features
        shap.summary_plot(
            self.shap_values, 
            feature_df,
            plot_type="dot",  # Try dot plot which shows distribution better
            show=False,
            max_display=n_features,  # Explicitly set max_display
            sort=True  # Ensure features are sorted by importance
        )

        plt.title("SHAP Summary Plot\nFeatures ranked by importance; Color shows feature value")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Enhanced SHAP analysis
        print(f"Figures saved in {output_dir}/:")
        print("- statistical_metrics.png")
        print("- pca_components.png")
        print("- feature_correlations.png")
        print("- shap_importance.png")
        print("- shap_summary.png")
        
        return self._generate_feature_summary(n_features)
    
    def _generate_feature_summary(self, n_features):
        """Generate detailed summary of top features for each metric"""
        metrics_explanation = {
            'mutual_info': """
                Mutual Information Score:
                - Measures any kind of statistical dependency between features and target
                - Range: 0 (independent) to 1 (strong dependency)
                - Can capture non-linear relationships
                - Not affected by scale of the features
            """,
            'f_score': """
                F-Score:
                - Measures strength of linear relationships
                - Higher values indicate stronger linear correlation
                - Based on ratio of variances (ANOVA)
                - Best for detecting linear relationships
            """,
            'spearman_corr': """
                Spearman Correlation:
                - Measures monotonic relationships (consistent increase/decrease)
                - Range: -1 to 1 (negative values indicate inverse relationships)
                - More robust to outliers than linear correlation
                - Good for ordinal relationships
                - Features ranked by absolute value but showing actual correlation
            """,
            'pca_importance': """
                PCA Loading:
                - Measures feature contribution to principal components
                - Higher values indicate more variance explained
                - Shows which features drive most variation in data
                - Useful for dimensionality reduction
            """,
            'shap_importance': """
                SHAP Values:
                - Measures feature impact on model predictions
                - Based on game theory concepts
                - Shows both magnitude and direction of impact
                - Considers feature interactions
            """
        }
        
        summary = {}
        for metric in metrics_explanation.keys():
            if metric == 'spearman_corr':
                # For Spearman, get top features by absolute value but show actual correlation
                top_feat = self.feature_importances.nlargest(n_features, 'spearman_corr_rank')
                summary[metric] = {
                    'explanation': metrics_explanation[metric],
                    'top_features': [
                        f"{row['feature']} (correlation: {row[metric]:.4f})"
                        for _, row in top_feat.iterrows()
                    ]
                }
            else:
                # Original handling for other metrics
                top_feat = self.feature_importances.nlargest(n_features, metric)
                summary[metric] = {
                    'explanation': metrics_explanation[metric],
                    'top_features': [
                        f"{row['feature']} (score: {row[metric]:.4f})"
                        for _, row in top_feat.iterrows()
                    ]
                }
        
        return summary

class WatermelonDataset(Dataset):
    """Custom Dataset for watermelon features"""
    def __init__(self, X, y):
        # Convert to numpy if pandas Series/DataFrame
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BrixPredictor(nn.Module):
    """Neural network for Brix score prediction"""
    def __init__(self, input_dim):
        super(BrixPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def ensure_1d(arr):
    """Ensure array is 1-dimensional"""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        return arr.squeeze()
    return arr

def main():
    for type in ['sweep', 'tap', 'impulse']:
        # Initialize and run analysis
        explorer = FeatureExplorer(
            features_csv=f'data/{type}_audio_features.csv',
            metadata_csv='data/metadata.csv'
        )
        
        # Prepare data and analyze
        merged_data = explorer.prepare_data()
        explorer.analyze_feature_importance()

        output_dir = f'data/figures/{type}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save plots
        print("Generating and saving feature analysis plots...")
        feature_summary = explorer.plot_feature_analysis(n_features=10, output_dir=output_dir)
        
        # Print detailed analysis
        print("\nDetailed Feature Analysis:")
        for metric, info in feature_summary.items():
            print(f"\n{metric.upper()}")
            print(info['explanation'])
            print("\nTop Features:")
            for i, feat in enumerate(info['top_features'], 1):
                print(f"{i}. {feat}")
        
        # Save feature importance data
        explorer.feature_importances.to_csv(f'{output_dir}/{type}_feature_importance.csv', index=False)
        print(f"\nAnalysis complete. Results saved to '{output_dir}/{type}_feature_importance.csv'")

if __name__ == "__main__":
    main()
