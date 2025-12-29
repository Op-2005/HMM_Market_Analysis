# Data loader for financial time series with normalization and regime labeling.
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .feature_engineering import add_technical_indicators

class FinancialDataLoader(Dataset):
    def __init__(self, file_path, target_column, features, normalize=True, data=None, device=None):
        self.device = 'cpu'
        self.target_column = target_column[0] if isinstance(target_column, (list, tuple)) else target_column
        self.features = features
        self.normalize = normalize
        
        # Load data from file or use provided data
        if data is None:
            self.data = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}, shape: {self.data.shape}")
        else:
            self.data = data.copy()
        
        self.data.columns = self.data.columns.str.strip()  # Remove whitespace from column names
        
        # Remove rows with missing values in target or feature columns
        subset_cols = [self.target_column] + features
        subset_cols = [col for col in subset_cols if col in self.data.columns]
        original_len = len(self.data)
        self.data.dropna(subset=subset_cols, inplace=True)
        if original_len - len(self.data) > 0:
            print(f"Dropped {original_len - len(self.data)} rows with NaN values")
        
        self.X = self.data[features].values.astype(np.float32)
        self.y = self.data[self.target_column].values.astype(np.float32)
        
        # Normalize features to zero mean and unit variance
        if self.normalize:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
            self.X = (self.X - self.mean) / self.std
            print(f"Normalized features, mean: {self.mean}, std: {self.std}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float32)
        return X_tensor, y_tensor

    def get_data_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def train_test_split(self, test_size=0.2, shuffle=True):
        num_samples = len(self.data)
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        split_idx = int(num_samples * (1 - test_size))
        train_data = self.data.iloc[indices[:split_idx]].copy()
        test_data = self.data.iloc[indices[split_idx:]].copy()
        
        train_loader = FinancialDataLoader(
            file_path=None, target_column=self.target_column, features=self.features, 
            normalize=self.normalize, data=train_data, device=self.device
        )
        test_loader = FinancialDataLoader(
            file_path=None, target_column=self.target_column, features=self.features, 
            normalize=self.normalize, data=test_data, device=self.device
        )
        return train_loader, test_loader

    def add_log_returns(self, price_column):
        # Compute log returns: log(price_t / price_{t-1})
        if price_column not in self.data.columns:
            raise ValueError(f"Column {price_column} not found in data")
        
        log_returns_col = f"{price_column}_log_return"
        self.data[log_returns_col] = np.log(self.data[price_column] / self.data[price_column].shift(1))
        self.data.dropna(subset=[log_returns_col], inplace=True)  # First row will be NaN
        print(f"Added log returns column: {log_returns_col}")
        return log_returns_col

    def add_regime_labels(self, returns_column, threshold=0.0, window=None):
        # Create binary labels: 1 = bull market (returns > threshold), 0 = bear market
        if returns_column not in self.data.columns:
            raise ValueError(f"Column {returns_column} not found in data")
        
        label_col = "actual_label"
        
        if window is not None and window > 1:
            # Use smoothed returns over a rolling window to reduce noise
            smoothed_returns = self.data[returns_column].rolling(window=window).mean()
            self.data[label_col] = (smoothed_returns > threshold).astype(int)
            print(f"Added regime labels using {window}-day smoothed returns")
        else:
            # Use raw returns for labeling
            self.data[label_col] = (self.data[returns_column] > threshold).astype(int)
            print(f"Added regime labels using daily returns with threshold {threshold}")
        
        self.data.dropna(subset=[label_col], inplace=True)
        bull_count = self.data[label_col].sum()
        bear_count = (self.data[label_col] == 0).sum()
        print(f"Added regime labels column: {label_col}")
        print(f"Bull market days: {bull_count} ({bull_count/len(self.data)*100:.1f}%)")
        print(f"Bear market days: {bear_count} ({bear_count/len(self.data)*100:.1f}%)")
        return label_col
    
    def add_technical_indicators(self, price_col, high_col=None, low_col=None, volume_col=None,
                                rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9,
                                bb_window=20, bb_std=2.0, atr_window=14, volume_window=20):
        original_cols = set(self.data.columns)
        self.data = add_technical_indicators(
            self.data, price_col=price_col, high_col=high_col, low_col=low_col, volume_col=volume_col,
            rsi_window=rsi_window, macd_fast=macd_fast, macd_slow=macd_slow, macd_signal=macd_signal,
            bb_window=bb_window, bb_std=bb_std, atr_window=atr_window, volume_window=volume_window
        )
        added_cols = [col for col in self.data.columns if col not in original_cols]
        print(f"Added {len(added_cols)} technical indicator columns: {', '.join(added_cols)}")
        return added_cols

class Discretizer:
    # Discretizes continuous data into bins. Fits on training data and transforms test data to prevent data leakage.
    def __init__(self, num_bins=10, strategy='equal_width', random_state=0, 
                 multivariate_method='independent', clip_outliers=True, outlier_percentile=99):
        self.num_bins = num_bins
        self.strategy = strategy
        self.random_state = random_state
        self.multivariate_method = multivariate_method
        self.clip_outliers = clip_outliers
        self.outlier_percentile = outlier_percentile
        self.bins_ = None
        self.kmeans_model_ = None
        self.pca_model_ = None
        self.fitted_ = False
        self.is_multivariate_ = False
    
    def fit(self, data):
        # Fit discretizer on training data to learn bin boundaries
        data = np.array(data)
        
        # Handle multivariate data
        if len(data.shape) == 2 and data.shape[1] > 1:
            self.is_multivariate_ = True
            if self.multivariate_method == 'pca_kmeans':
                return self._fit_multivariate_pca_kmeans(data)
            else:
                return self._fit_multivariate_independent(data)
        
        self.is_multivariate_ = False
        if len(data.shape) > 1:
            data = data.flatten()
        
        # Clip outliers to prevent extreme values from affecting bin boundaries
        if self.clip_outliers:
            lower = np.percentile(data, 100 - self.outlier_percentile)
            upper = np.percentile(data, self.outlier_percentile)
            data = np.clip(data, lower, upper)
            self.clip_bounds_ = (lower, upper)
        else:
            self.clip_bounds_ = None
        
        # Compute bin boundaries based on strategy
        if self.strategy == 'equal_width':
            # Equal width bins: divide range into equal-sized intervals
            self.bins_ = np.linspace(np.min(data), np.max(data), self.num_bins + 1)
        elif self.strategy == 'equal_freq':
            # Equal frequency bins: each bin contains roughly same number of samples
            self.bins_ = np.percentile(data, np.linspace(0, 100, self.num_bins + 1))
        elif self.strategy == 'kmeans':
            # K-means clustering: bins based on data clusters
            from sklearn.cluster import KMeans
            self.kmeans_model_ = KMeans(n_clusters=self.num_bins, random_state=self.random_state)
            self.kmeans_model_.fit(data.reshape(-1, 1))
            centers = np.sort(self.kmeans_model_.cluster_centers_.flatten())
            self.bins_ = np.concatenate([[-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]])
        else:
            raise ValueError(f"Unknown discretization strategy: {self.strategy}")
        
        self.bins_ = np.unique(self.bins_)
        self.fitted_ = True
        return self
    
    def _fit_multivariate_independent(self, data):
        n_features = data.shape[1]
        self.discretizers_ = []
        for i in range(n_features):
            disc = Discretizer(
                num_bins=self.num_bins, strategy=self.strategy, random_state=self.random_state + i,
                clip_outliers=self.clip_outliers, outlier_percentile=self.outlier_percentile,
                multivariate_method='independent'
            )
            disc.fit(data[:, i])
            self.discretizers_.append(disc)
        self.fitted_ = True
        return self
    
    def _fit_multivariate_pca_kmeans(self, data):
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        
        n_components = min(3, data.shape[1])
        self.pca_model_ = PCA(n_components=n_components, random_state=self.random_state)
        data_pca = self.pca_model_.fit_transform(data)
        self.kmeans_model_ = KMeans(n_clusters=self.num_bins, random_state=self.random_state)
        self.kmeans_model_.fit(data_pca)
        self.fitted_ = True
        return self
    
    def transform(self, data):
        # Transform data using previously fitted bin boundaries (for test data)
        if not self.fitted_:
            raise ValueError("Discretizer must be fitted before transform. Call fit() first.")
        
        data = np.array(data)
        
        if self.is_multivariate_:
            if self.multivariate_method == 'pca_kmeans':
                return self._transform_multivariate_pca_kmeans(data)
            else:
                return self._transform_multivariate_independent(data)
        
        if len(data.shape) > 1:
            data = data.flatten()
        
        # Apply same outlier clipping that was used during fit
        if self.clip_outliers and self.clip_bounds_ is not None:
            lower, upper = self.clip_bounds_
            data = np.clip(data, lower, upper)
        
        # Assign each value to a bin
        if self.strategy == 'kmeans' and self.kmeans_model_ is not None:
            discretized = self.kmeans_model_.predict(data.reshape(-1, 1))
            return discretized.reshape(data.shape)
        else:
            discretized = np.digitize(data, self.bins_) - 1  # Find bin index for each value
            discretized = np.clip(discretized, 0, self.num_bins - 1)  # Ensure valid bin indices
            return discretized.reshape(data.shape)
    
    def _transform_multivariate_independent(self, data):
        n_features = data.shape[1]
        discretized_list = []
        for i in range(n_features):
            disc = self.discretizers_[i]
            discretized_list.append(disc.transform(data[:, i]))
        return np.column_stack(discretized_list)
    
    def _transform_multivariate_pca_kmeans(self, data):
        data_pca = self.pca_model_.transform(data)
        discretized = self.kmeans_model_.predict(data_pca)
        return discretized
    
    def fit_transform(self, data):
        return self.fit(data).transform(data)

def discretize_data(data, num_bins=10, strategy='equal_width'):
    # Convenience function for discretization. Use Discretizer class for train/test splits to prevent data leakage.
    if len(data.shape) > 1 and data.shape[1] > 1:
        raise ValueError("discretize_data expects a 1D array")
    
    flat_data = data.flatten()
    
    if strategy == 'equal_width':
        bins = np.linspace(np.min(flat_data), np.max(flat_data), num_bins + 1)
    elif strategy == 'equal_freq':
        bins = np.percentile(flat_data, np.linspace(0, 100, num_bins + 1))
    elif strategy == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(flat_data.reshape(-1, 1))
        centers = np.sort(kmeans.cluster_centers_.flatten())
        bins = np.concatenate([[-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]])
    else:
        raise ValueError(f"Unknown discretization strategy: {strategy}")
    
    bins = np.unique(bins)
    discretized = np.digitize(flat_data, bins) - 1
    discretized = np.minimum(discretized, num_bins - 1)
    return discretized.reshape(data.shape)
