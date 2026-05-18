import numpy as np

# REGRESSION MODELS
class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        if self.fit_intercept:
            X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        else:
            X_with_intercept = X
        
        # Normal Equation: β = (X^T X)^(-1) X^T y
        # Use solve instead of inv to avoid numerical issues
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        weights = np.linalg.solve(XtX, Xty)
        
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_

# LASSO 
class Lasso:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
    
    def _soft_threshold(self, rho, alpha):
        return np.sign(rho) * np.maximum(np.abs(rho) - alpha, 0)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            X_with_intercept = np.c_[np.ones(n_samples), X]
        else:
            X_with_intercept = X
        
        weights = np.zeros(X_with_intercept.shape[1])
        
        # Pre-compute X^T @ X diagonal for optimization
        XtX_diag = np.sum(X_with_intercept ** 2, axis=0)
        
        # Coordinate Descent
        for iteration in range(self.max_iter):
            weights_old = weights.copy()
            
            # Vectorized update for all weights
            for j in range(len(weights)):
                # Calculate residual excluding feature j
                residual = y - X_with_intercept @ weights + weights[j] * X_with_intercept[:, j]
                
                # Calculate rho_j = X_j^T * residual
                rho = X_with_intercept[:, j] @ residual
                
                # Update weight j
                if j == 0 and self.fit_intercept:
                    weights[j] = rho / n_samples
                else:
                    # Soft-thresholding with alpha
                    weights[j] = self._soft_threshold(rho / n_samples, self.alpha) / (XtX_diag[j] / n_samples)
            
            # Check convergence
            if np.sum(np.abs(weights - weights_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        if self.fit_intercept:
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights
        
        return self
    
    def predict(self, X):
        return X @ self.coef_ + self.intercept_


# CART (Classification and Regression Tree)
class CART:
    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None
        self.n_features_ = None
        self.n_leaves_ = 0
    
    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None, 0
        
        current_mse = self._mse(y)
        best_mse_reduction = 0
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                mse_left = self._mse(y_left)
                mse_right = self._mse(y_right)
                
                weighted_mse = (n_left * mse_left + n_right * mse_right) / n_samples
                mse_reduction = current_mse - weighted_mse

                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_mse_reduction
    
    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf or
            len(np.unique(y)) == 1):
            self.n_leaves_ += 1
            return {
                'type': 'leaf',
                'value': np.mean(y),
                'n_samples': n_samples
            }
        
        best_feature, best_threshold, mse_reduction = self._best_split(X, y)
        
        if best_feature is None or mse_reduction <= 0:
            self.n_leaves_ += 1
            return {
                'type': 'leaf',
                'value': np.mean(y),
                'n_samples': n_samples
            }
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'type': 'internal',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child,
            'n_samples': n_samples,
            'mse_reduction': mse_reduction
        }
    
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.n_leaves_ = 0
        self.tree_ = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        if node['type'] == 'leaf':
            return node['value']
        
        # Navigate tree
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree_) for x in X])
    
    def get_n_leaves(self):
        return self.n_leaves_
    
    def get_depth(self):
        def _get_depth(node):
            if node['type'] == 'leaf':
                return 0
            return 1 + max(_get_depth(node['left']), _get_depth(node['right']))
        
        return _get_depth(self.tree_)


# CROSS-VALIDATION
class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        # Vectorized split calculation
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, test_idx
            current = stop


# METRICS
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot)


def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def evaluate_model(y_true, y_pred, model_name="Model"):
    r2 = r2_score(y_true, y_pred)
    rmse = rmse_score(y_true, y_pred)
    mae = mae_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE: {mae:.3f}")
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


def cross_validate(estimator, X, y, cv=5):
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    
    print(f"\nPerforming {cv.n_splits if hasattr(cv, 'n_splits') else 'K'}-Fold Cross-Validation...")
    print(f"Model: {type(estimator).__name__}" + 
          (f" (alpha={estimator.alpha})" if hasattr(estimator, 'alpha') else ""))
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        if hasattr(estimator, 'alpha'):
            fold_estimator = type(estimator)(alpha=estimator.alpha, 
                                             max_iter=estimator.max_iter, 
                                             tol=estimator.tol)
        else:
            fold_estimator = type(estimator)()
        
        fold_estimator.fit(X_fold_train, y_fold_train)

        y_fold_pred = fold_estimator.predict(X_fold_val)

        r2 = r2_score(y_fold_val, y_fold_pred)
        rmse = rmse_score(y_fold_val, y_fold_pred)
        mae = mae_score(y_fold_val, y_fold_pred)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"  Fold {fold_idx + 1}: R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

    r2_scores = np.array(r2_scores)
    rmse_scores = np.array(rmse_scores)
    mae_scores = np.array(mae_scores)
    
    print(f"\n  Mean R²: {r2_scores.mean():.3f} (+/- {r2_scores.std():.4f})")
    print(f"  Mean RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std():.4f})")
    print(f"  Mean MAE: {mae_scores.mean():.3f} (+/- {mae_scores.std():.4f})")
    
    return {
        'r2_scores': r2_scores,
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores,
        'mean_r2': r2_scores.mean(),
        'std_r2': r2_scores.std(),
        'mean_rmse': rmse_scores.mean(),
        'std_rmse': rmse_scores.std(),
        'mean_mae': mae_scores.mean(),
        'std_mae': mae_scores.std()
    }
