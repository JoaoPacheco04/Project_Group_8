import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class SimpleSVD:
    """A simple Matrix Factorization (Funk SVD) implementation to replace scikit-surprise
    when it cannot be installed (e.g., on Python 3.13 Windows).
    """
    def __init__(self, n_factors=20, n_epochs=20, lr=0.005, reg=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.global_mean = 0
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}

    def fit(self, df):
        """Fit the model on a dataframe with columns: [user, item, rating]"""
        # Ensure correct column order
        cols = df.columns
        user_col, item_col, rating_col = cols[0], cols[1], cols[2]
        
        # Create mappings
        unique_users = df[user_col].unique()
        unique_items = df[item_col].unique()
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {item: i for i, item in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Initialize factors
        rng = np.random.default_rng(self.random_state)
        self.user_factors = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (n_items, self.n_factors))
        
        self.global_mean = df[rating_col].mean()
        
        # Convert df to numpy for speed
        u_indices = df[user_col].map(self.user_map).values
        i_indices = df[item_col].map(self.item_map).values
        ratings = df[rating_col].values
        
        print(f"Training SimpleSVD on {len(df)} ratings...")
        for epoch in range(self.n_epochs):
            for i in range(len(ratings)):
                u, it, r = u_indices[i], i_indices[i], ratings[i]
                
                # Predict
                pred = self.global_mean + np.dot(self.user_factors[u], self.item_factors[it])
                err = r - pred
                
                # Update factors
                u_f = self.user_factors[u]
                it_f = self.item_factors[it]
                
                self.user_factors[u] += self.lr * (err * it_f - self.reg * u_f)
                self.item_factors[it] += self.lr * (err * u_f - self.reg * it_f)
            
            if (epoch + 1) % 5 == 0:
                print(f" Epoch {epoch+1}/{self.n_epochs} complete")

    def predict(self, user, item):
        """Predict the rating for a given user and item. 
        Returns an object with an 'est' attribute for compatibility with Surprise.
        """
        u_idx = self.user_map.get(user)
        i_idx = self.item_map.get(item)
        
        if u_idx is None or i_idx is None:
            res = self.global_mean
        else:
            res = self.global_mean + np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
            
        # Return object with .est for compatibility
        return type('obj', (object,), {'est': res})

def simple_cross_validate(df, n_factors=20, cv=5):
    """Simple K-Fold cross validation for the SimpleSVD model."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    rmses = []
    maes = []
    
    fold = 1
    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        model = SimpleSVD(n_factors=n_factors)
        model.fit(train_df)
        
        preds = test_df.apply(lambda x: model.predict(x.iloc[0], x.iloc[1]).est, axis=1)
        actuals = test_df.iloc[:, 2]
        
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = np.mean(np.abs(actuals - preds))
        
        rmses.append(rmse)
        maes.append(mae)
        print(f"Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}")
        fold += 1
        
    return {"test_rmse": np.array(rmses), "test_mae": np.array(maes)}


def get_top_n_recommendations(model, user_id, df_original, n=10):
    """Return top-N unseen items for a given user."""
    user_col, item_col = df_original.columns[0], df_original.columns[1]
    all_items = df_original[item_col].unique()
    rated_items = df_original[df_original[user_col] == user_id][item_col].tolist()

    predictions = []
    for item_id in all_items:
        if item_id in rated_items:
            continue
        pred = model.predict(user_id, item_id)
        predictions.append((item_id, pred.est))

    return sorted(predictions, key=lambda row: row[1], reverse=True)[:n]
