import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .feature_engineering import add_technical_indicators

class FinancialDataLoader(Dataset):
    def __init__(self, file_path, target_column, features, normalize=True, data=None, device=None):
        self.device = 'cpu'
        
        if isinstance(target_column, (list, tuple)):
            self.target_column = target_column[0]
        else:
            self.target_column = target_column
        
        self.features = features
        self.normalize = normalize

        if data is None:
            self.data = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}, shape: {self.data.shape}")
        else:
            self.data = data.copy()

        self.data.columns = self.data.columns.str.strip()

        subset_cols = [self.target_column] + features
        subset_cols = [col for col in subset_cols if col in self.data.columns]
        original_len = len(self.data)
        self.data.dropna(subset=subset_cols, inplace=True)
        dropped_rows = original_len - len(self.data)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with NaN values")

        self.X = self.data[features].values.astype(np.float32)
        self.y = self.data[self.target_column].values.astype(np.float32)

        if self.normalize:
            self.mean = np.mean(self.X, axis=0)
            self.std = np.std(self.X, axis=0) + 1e-8
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
        print(f"Splitting data: {num_samples} samples with test_size={test_size}")

        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)

        split_idx = int(num_samples * (1 - test_size))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        train_data = self.data.iloc[train_indices].copy()
        test_data = self.data.iloc[test_indices].copy()

        print(f"Train set: {len(train_data)} samples, Test set: {len(test_data)} samples")

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
        if price_column not in self.data.columns:
            raise ValueError(f"Column {price_column} not found in data")
        
        log_returns_col = f"{price_column}_log_return"
        self.data[log_returns_col] = np.log(self.data[price_column] / self.data[price_column].shift(1))
        self.data.dropna(subset=[log_returns_col], inplace=True)
        
        print(f"Added log returns column: {log_returns_col}")
        return log_returns_col

    def add_regime_labels(self, returns_column, threshold=0.0, window=None):
        if returns_column not in self.data.columns:
            raise ValueError(f"Column {returns_column} not found in data")
        
        label_col = "actual_label"
        
        if window is not None and window > 1:
            smoothed_returns = self.data[returns_column].rolling(window=window).mean()
            self.data[label_col] = (smoothed_returns > threshold).astype(int)
            print(f"Added regime labels using {window}-day smoothed returns")
        else:
            self.data[label_col] = (self.data[returns_column] > threshold).astype(int)
            print(f"Added regime labels using daily returns with threshold {threshold}")
        
        self.data.dropna(subset=[label_col], inplace=True)
        
        print(f"Added regime labels column: {label_col}")
        print(f"Bull market days: {self.data[label_col].sum()} ({self.data[label_col].mean()*100:.1f}%)")
        print(f"Bear market days: {(self.data[label_col] == 0).sum()} ({(1-self.data[label_col].mean())*100:.1f}%)")
        
        return label_col
    
    def add_technical_indicators(self, price_col, high_col=None, low_col=None, volume_col=None,
                                rsi_window=14, macd_fast=12, macd_slow=26, macd_signal=9,
                                bb_window=20, bb_std=2.0, atr_window=14, volume_window=20):
        """
        Add technical indicators to the dataset.
        
        Parameters:
        -----------
        price_col : str
            Column name for price/close
        high_col : str, optional
            Column name for high prices (required for ATR)
        low_col : str, optional
            Column name for low prices (required for ATR)
        volume_col : str, optional
            Column name for volume (required for volume ratio)
        rsi_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std, 
        atr_window, volume_window : int/float
            Parameters for technical indicators
        
        Returns:
        --------
        list
            List of added indicator column names
        """
        added_cols = []
        original_cols = set(self.data.columns)
        
        self.data = add_technical_indicators(
            self.data,
            price_col=price_col,
            high_col=high_col,
            low_col=low_col,
            volume_col=volume_col,
            rsi_window=rsi_window,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_window=bb_window,
            bb_std=bb_std,
            atr_window=atr_window,
            volume_window=volume_window
        )
        
        added_cols = [col for col in self.data.columns if col not in original_cols]
        print(f"Added {len(added_cols)} technical indicator columns: {', '.join(added_cols)}")
        
        return added_cols


class Discretizer:
    """
    Fits discretization on training data and applies to test data to prevent data leakage.
    
    Supports both univariate and multivariate data discretization.
    """
    def __init__(self, num_bins=10, strategy='equal_width', random_state=0, 
                 multivariate_method='independent', clip_outliers=True, outlier_percentile=99):
        """
        Initialize Discretizer.
        
        Parameters:
        -----------
        num_bins : int
            Number of bins for discretization
        strategy : str
            Discretization strategy: 'equal_width', 'equal_freq', 'kmeans'
        random_state : int
            Random seed for reproducibility
        multivariate_method : str
            Method for multivariate data: 'independent', 'pca_kmeans'
        clip_outliers : bool
            Whether to clip outliers before discretization
        outlier_percentile : float
            Percentile threshold for outlier clipping (e.g., 99 for 1st and 99th percentile)
        """
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
        """
        Fit discretization parameters on training data.
        
        Parameters:
        -----------
        data : array-like, 1D or 2D
            Training data to fit discretization on
            - 1D: Single feature (shape: [n_samples])
            - 2D: Multiple features (shape: [n_samples, n_features])
        """
        data = np.array(data)
        
        # Handle multivariate case
        if len(data.shape) == 2 and data.shape[1] > 1:
            self.is_multivariate_ = True
            if self.multivariate_method == 'pca_kmeans':
                return self._fit_multivariate_pca_kmeans(data)
            else:  # independent
                return self._fit_multivariate_independent(data)
        
        # Univariate case
        self.is_multivariate_ = False
        if len(data.shape) > 1:
            data = data.flatten()
        
        # Clip outliers if requested
        if self.clip_outliers:
            lower = np.percentile(data, 100 - self.outlier_percentile)
            upper = np.percentile(data, self.outlier_percentile)
            data = np.clip(data, lower, upper)
            self.clip_bounds_ = (lower, upper)
        else:
            self.clip_bounds_ = None
        
        if self.strategy == 'equal_width':
            self.bins_ = np.linspace(np.min(data), np.max(data), self.num_bins + 1)
        elif self.strategy == 'equal_freq':
            self.bins_ = np.percentile(data, np.linspace(0, 100, self.num_bins + 1))
        elif self.strategy == 'kmeans':
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
        """Fit independent discretizers for each feature."""
        n_features = data.shape[1]
        self.discretizers_ = []
        
        for i in range(n_features):
            disc = Discretizer(
                num_bins=self.num_bins,
                strategy=self.strategy,
                random_state=self.random_state + i,
                clip_outliers=self.clip_outliers,
                outlier_percentile=self.outlier_percentile,
                multivariate_method='independent'  # Prevent recursion
            )
            disc.fit(data[:, i])
            self.discretizers_.append(disc)
        
        self.fitted_ = True
        return self
    
    def _fit_multivariate_pca_kmeans(self, data):
        """Fit PCA + KMeans for multivariate discretization."""
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        
        # Reduce dimensions first
        n_components = min(3, data.shape[1])
        self.pca_model_ = PCA(n_components=n_components, random_state=self.random_state)
        data_pca = self.pca_model_.fit_transform(data)
        
        # Then use KMeans on reduced dimensions
        self.kmeans_model_ = KMeans(
            n_clusters=self.num_bins, 
            random_state=self.random_state
        )
        self.kmeans_model_.fit(data_pca)
        
        self.fitted_ = True
        return self
    
    def transform(self, data):
        """
        Apply fitted discretization to new data.
        
        Parameters:
        -----------
        data : array-like, 1D or 2D
            Data to discretize using fitted bins
        
        Returns:
        --------
        discretized : ndarray
            Discretized data with same shape as input (or combined shape for multivariate)
        """
        if not self.fitted_:
            raise ValueError("Discretizer must be fitted before transform. Call fit() first.")
        
        data = np.array(data)
        
        # Handle multivariate case
        if self.is_multivariate_:
            if self.multivariate_method == 'pca_kmeans':
                return self._transform_multivariate_pca_kmeans(data)
            else:  # independent
                return self._transform_multivariate_independent(data)
        
        # Univariate case
        if len(data.shape) > 1:
            data = data.flatten()
        
        # Clip outliers if it was done during fit
        if self.clip_outliers and self.clip_bounds_ is not None:
            lower, upper = self.clip_bounds_
            data = np.clip(data, lower, upper)
        
        if self.strategy == 'kmeans' and self.kmeans_model_ is not None:
            discretized = self.kmeans_model_.predict(data.reshape(-1, 1))
            return discretized.reshape(data.shape)
        else:
            discretized = np.digitize(data, self.bins_) - 1
            discretized = np.clip(discretized, 0, self.num_bins - 1)
            return discretized.reshape(data.shape)
    
    def _transform_multivariate_independent(self, data):
        """Transform multivariate data using independent discretizers."""
        n_features = data.shape[1]
        discretized_list = []
        
        for i in range(n_features):
            disc = self.discretizers_[i]
            disc_result = disc.transform(data[:, i])
            discretized_list.append(disc_result)
        
        # Return combined result (can be used with feature combination strategies)
        return np.column_stack(discretized_list)
    
    def _transform_multivariate_pca_kmeans(self, data):
        """Transform multivariate data using PCA + KMeans."""
        data_pca = self.pca_model_.transform(data)
        discretized = self.kmeans_model_.predict(data_pca)
        return discretized
    
    def fit_transform(self, data):
        """Fit discretizer and transform data in one step."""
        return self.fit(data).transform(data)


def discretize_data(data, num_bins=10, strategy='equal_width'):
    """
    Convenience function for discretization. 
    
    WARNING: For train/test splits, use Discretizer class instead to prevent data leakage.
    This function fits on the provided data, which causes leakage if used separately on train/test.
    
    For proper usage:
        discretizer = Discretizer(num_bins, strategy)
        train_discrete = discretizer.fit_transform(train_data)
        test_discrete = discretizer.transform(test_data)
    """
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


def combine_features(data, features, method='first'):
    if method == 'first':
        if isinstance(data, pd.DataFrame):
            return data[features[0]].values
        else:
            return data[:, 0]
    elif method == 'mean':
        if isinstance(data, pd.DataFrame):
            return data[features].mean(axis=1).values
        else:
            return np.mean(data, axis=1)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        if isinstance(data, pd.DataFrame):
            return pca.fit_transform(data[features].values).flatten()
        else:
            return pca.fit_transform(data).flatten()
    elif method == 'custom':
        pass
    else:
        raise ValueError(f"Unknown feature combination method: {method}")


def map_bins_to_values(discretized_data, original_data, strategy='midpoint'):
    unique_bins = np.unique(discretized_data)
    bin_map = {}
    
    for bin_idx in unique_bins:
        bin_mask = (discretized_data == bin_idx)
        bin_data = original_data[bin_mask]
        
        if strategy == 'midpoint':
            bin_map[bin_idx] = (np.min(bin_data) + np.max(bin_data)) / 2
        elif strategy == 'mean':
            bin_map[bin_idx] = np.mean(bin_data)
        else:
            raise ValueError(f"Unknown mapping strategy: {strategy}")
    
    return np.array([bin_map[idx] for idx in discretized_data.flatten()]).reshape(discretized_data.shape)
