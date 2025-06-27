#date: 2025-06-27T17:09:33Z
#url: https://api.github.com/gists/6fd27560910d8d6fcb9567ca9974163b
#owner: https://api.github.com/users/dividenconquer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class NoShowPredictor:
    def __init__(self, friction_cost=100, avg_lost_value=1000, false_negative_penalty=4):
        """
        Initialize the No-Show Predictor.
        
        Args:
            friction_cost: Cost of requesting deposit - 100원
            avg_lost_value: Revenue lost per no-show - 1000원 
            false_negative_penalty: Weight for false negative errors
        """
        self.friction_cost = friction_cost
        self.avg_lost_value = avg_lost_value
        self.false_negative_penalty = false_negative_penalty
        self.scaler = StandardScaler()
        self.models = {}
        self.calibrated_models = {}
        self.feature_names = None
        

    
    def load_and_prepare_data(self, csv_file='booking_history_sample.csv'):
        """Load data from CSV file and prepare features."""
        print(f"Loading data from {csv_file}...")
        
        # Load CSV data
        data = pd.read_csv(csv_file)
        
        print(f"Loaded {len(data)} records from CSV")
        print("Data columns:", data.columns.tolist())
        
        # Convert date columns
        data['created_at'] = pd.to_datetime(data['created_at'])
        data['first_date'] = pd.to_datetime(data['first_date'])
        
        # Create target variable (success = False means no-show)
        data['no_show'] = ~data['success']
        
        # Calculate lead time (days between booking and reservation)
        data['lead_time'] = (data['first_date'] - data['created_at']).dt.days
        
        # Extract time features
        data['hour'] = pd.to_datetime(data['first_time'], format='%H:%M').dt.hour
        data['weekday'] = data['first_date'].dt.day_name()
        data['month'] = data['first_date'].dt.month
        data['is_weekend'] = data['first_date'].dt.weekday >= 5
        
        # Party size features
        data['party_size'] = data['total_count']
        data['has_children'] = (data['child_count'] > 0) | (data['infant_count'] > 0)
        data['large_party'] = data['party_size'] >= 6
        
        # Time slot categorization
        def categorize_time_slot(hour):
            if 9 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 15:
                return 'Lunch'
            elif 15 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 21:
                return 'Dinner'
            else:
                return 'Late'
                
        data['time_slot'] = data['hour'].apply(categorize_time_slot)
        
        # Season categorization
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
                
        data['season'] = data['month'].apply(get_season)
        
        # Historical behavior features
        data['no_show_rate'] = np.where(data['past_reservations'] > 0, 
                                       data['past_no_shows'] / data['past_reservations'], 0)
        
        # Advanced booking ratio
        data['advance_booking_ratio'] = data['lead_time'] / (data['lead_time'] + 1)
        
        # Weather data is now included in the CSV file
        print("Using weather data from CSV file...")
        
        # Fill any missing weather data (if any)
        if 'weather' not in data.columns:
            print("Warning: No weather column found in CSV. Adding default weather data.")
            data['weather'] = 'Clear'
        else:
            data['weather'].fillna('Clear', inplace=True)
        
        return data
    
    def prepare_features(self, data):
        """Prepare features for modeling."""
        df = data.copy()
        
        # Handle missing values
        df['past_no_shows'].fillna(df['past_no_shows'].median(), inplace=True)
        df['past_reservations'].fillna(1, inplace=True)
        
        # Ensure no_show_rate is calculated correctly
        df['no_show_rate'] = np.where(df['past_reservations'] > 0, 
                                     df['past_no_shows'] / df['past_reservations'], 0)
        
        # Select features for modeling (excluding capacity and reservation_method)
        feature_columns = [
            'lead_time', 'party_size', 'past_no_shows', 'past_reservations', 
            'no_show_rate', 'advance_booking_ratio', 'hour', 'month',
            'adult_count', 'child_count', 'infant_count'
        ]
        
        # Categorical features to encode
        categorical_cols = ['weekday', 'time_slot', 'weather', 'season', 'name']
        
        # Create feature matrix
        X_numeric = df[feature_columns]
        X_categorical = pd.get_dummies(df[categorical_cols], drop_first=True)
        
        # Binary features
        X_binary = df[['is_weekend', 'has_children', 'large_party']].astype(int)
        
        # Combine all features
        X = pd.concat([X_numeric, X_categorical, X_binary], axis=1)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Normalize numerical features
        numerical_cols = feature_columns + ['hour', 'month']
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        y = df['no_show'].astype(int)
        
        return X, y, df
    
    def train_models(self, X_train, y_train):
        """Train logistic regression and random forest models."""
        
        # Logistic Regression
        self.models['logistic'] = LogisticRegression(max_iter=1000, random_state=42)
        self.models['logistic'].fit(X_train, y_train)
        
        # Random Forest with class balancing
        # Calculate class weights to handle imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1] * self.false_negative_penalty}
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weight_dict,
            random_state=42
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # Calibrate probabilities using cross-validation
        for name, model in self.models.items():
            self.calibrated_models[name] = CalibratedClassifierCV(model, method='sigmoid', cv=3)
            self.calibrated_models[name].fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation including business metrics."""
        results = {}
        
        for name, model in self.calibrated_models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
            # Statistical metrics
            brier_score = brier_score_loss(y_test, y_pred_proba)
            log_loss_score = log_loss(y_test, y_pred_proba)
            
            # Business metrics
            expected_profit = self.calculate_expected_profit(y_test, y_pred_proba)
            
            # Confusion matrix for detailed analysis
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results[name] = {
                'brier_score': brier_score,
                'log_loss': log_loss_score,
                'expected_profit': expected_profit,
                'confusion_matrix': cm,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
                'probabilities': y_pred_proba
            }
        
        return results
    
    def cross_validate_models(self, X, y, cv_folds=5):
        """Perform cross-validation with business metrics."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}
        
        for name, model in self.models.items():
            # Calibrated model for CV
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            
            brier_scores = []
            profit_scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Fit and predict
                calibrated_model.fit(X_train_cv, y_train_cv)
                y_pred_proba = calibrated_model.predict_proba(X_val_cv)[:, 1]
                
                # Calculate metrics
                brier_scores.append(brier_score_loss(y_val_cv, y_pred_proba))
                profit_scores.append(self.calculate_expected_profit(y_val_cv, y_pred_proba))
            
            cv_results[name] = {
                'brier_score_mean': np.mean(brier_scores),
                'brier_score_std': np.std(brier_scores),
                'profit_mean': np.mean(profit_scores),
                'profit_std': np.std(profit_scores)
            }
        
        return cv_results
    
    def calculate_expected_profit(self, y_true, y_pred_proba, threshold=None):
        """Calculate expected profit based on business logic."""
        if threshold is None:
            threshold = self.friction_cost / self.avg_lost_value
        
        # Convert to numpy arrays for easier indexing
        y_true_arr = np.array(y_true)
        y_pred_proba_arr = np.array(y_pred_proba)
        
        # Decision: request deposit if P(no-show) * lost_value > friction_cost
        request_deposit = y_pred_proba_arr > threshold
        
        profit = 0
        for i in range(len(y_true_arr)):
            if request_deposit[i]:
                if y_true_arr[i] == 1:  # Correctly identified no-show
                    profit += (self.avg_lost_value - self.friction_cost)  # Saved revenue minus friction
                else:  # False positive
                    profit -= self.friction_cost  # Only friction cost
            else:
                if y_true_arr[i] == 1:  # Missed no-show
                    profit -= self.avg_lost_value  # Lost revenue
                # True negatives have no cost
        
        return profit / len(y_true_arr)  # Average profit per reservation
    
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Find threshold that maximizes expected profit."""
        thresholds = np.linspace(0.01, 0.99, 100)
        profits = []
        
        for threshold in thresholds:
            profit = self.calculate_expected_profit(y_true, y_pred_proba, threshold)
            profits.append(profit)
        
        optimal_idx = np.argmax(profits)
        return thresholds[optimal_idx], profits[optimal_idx]
    
    def benchmark_static_policies(self, y_test):
        """Benchmark against static no-deposit policies."""
        # Never request deposit
        never_deposit_profit = -np.sum(y_test) * self.avg_lost_value / len(y_test)
        
        # Always request deposit
        always_deposit_profit = (
            np.sum(y_test) * (self.avg_lost_value - self.friction_cost) - 
            np.sum(1 - y_test) * self.friction_cost
        ) / len(y_test)
        
        return {
            'never_deposit': never_deposit_profit,
            'always_deposit': always_deposit_profit
        }
    
    def generate_deployment_report(self, X_test, y_test, results, original_data):
        """Generate comprehensive analysis report."""
        print("="*80)
        print("RESTAURANT NO-SHOW PREDICTION - ANALYSIS REPORT")
        print("="*80)
        
        # Data Overview
        print(f"\nDATA OVERVIEW:")
        print(f"• Total reservations: {len(original_data)}")
        print(f"• No-show rate: {original_data['no_show'].mean():.2%}")
        print(f"• Test set size: {len(y_test)}")
        print(f"• Cities covered: {original_data['name'].nunique()}")
        print(f"• Date range: {original_data['first_date'].min()} to {original_data['first_date'].max()}")
        
        # Model Parameters
        print(f"\nMODEL PARAMETERS:")
        print(f"• Deposit request cost: {self.friction_cost}원")
        print(f"• Revenue per reservation: {self.avg_lost_value}원")
        print(f"• Decision threshold: P(no-show) > {self.friction_cost/self.avg_lost_value:.3f}")
        
        # Model Performance
        print(f"\nMODEL PERFORMANCE:")
        for name, metrics in results.items():
            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  • Brier Score (lower is better): {metrics['brier_score']:.4f}")
            print(f"  • Expected Profit per Reservation: {metrics['expected_profit']:.0f}원")
            print(f"  • Confusion Matrix:")
            print(f"    - True Negatives: {metrics['true_negatives']} | False Positives: {metrics['false_positives']}")
            print(f"    - False Negatives: {metrics['false_negatives']} | True Positives: {metrics['true_positives']}")
        
        # Find optimal thresholds
        print(f"\nOPTIMAL THRESHOLD ANALYSIS:")
        for name, metrics in results.items():
            optimal_threshold, optimal_profit = self.find_optimal_threshold(y_test, metrics['probabilities'])
            print(f"{name.replace('_', ' ').title()}:")
            print(f"  • Optimal Threshold: {optimal_threshold:.3f}")
            print(f"  • Expected Profit at Optimal: {optimal_profit:.0f}원")
        
        # Benchmark comparison
        static_benchmarks = self.benchmark_static_policies(y_test)
        print(f"\nSTATIC POLICY BENCHMARKS:")
        print(f"• Never request deposit: {static_benchmarks['never_deposit']:.0f}원 per reservation")
        print(f"• Always request deposit: {static_benchmarks['always_deposit']:.0f}원 per reservation")
        
        # Feature importance (for Random Forest)
        if 'random_forest' in self.models:
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['random_forest'].feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTOP 10 FEATURE IMPORTANCE (Random Forest):")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  • {row['feature']}: {row['importance']:.4f}")
        
        # City-wise analysis
        print(f"\nCITY-WISE NO-SHOW ANALYSIS:")
        city_stats = original_data.groupby('name')['no_show'].agg(['count', 'mean']).sort_values('mean', ascending=False)
        for city, stats in city_stats.head(10).iterrows():
            print(f"  • {city}: {stats['mean']:.2%} no-show rate ({stats['count']} bookings)")
    
    def print_model_summary(self, model_name='random_forest'):
        """Print summary of the selected model and decision criteria."""
        print(f"\nMODEL SUMMARY:")
        print(f"Selected model: {model_name}")
        print(f"Decision criteria: Request deposit if P(no-show) × {self.avg_lost_value}원 > {self.friction_cost}원")
        print(f"Threshold: P(no-show) > {self.friction_cost/self.avg_lost_value:.3f}")
        print(f"This means we request deposit when no-show probability exceeds {self.friction_cost/self.avg_lost_value:.1%}")

def main():
    """Main function to run the no-show prediction analysis."""
    # Initialize predictor with parameters
    predictor = NoShowPredictor(friction_cost=100, avg_lost_value=1000, false_negative_penalty=4)
    
    # Load and prepare data from CSV
    data = predictor.load_and_prepare_data('booking_history_sample.csv')
    
    print("Preparing features for modeling...")
    X, y, original_data = predictor.prepare_features(data)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Feature columns: {len(predictor.feature_names)}")
    print(f"No-show rate: {y.mean():.2%}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )
    
    print("Training models...")
    predictor.train_models(X_train, y_train)
    
    print("Performing cross-validation...")
    cv_results = predictor.cross_validate_models(X, y)
    
    print("\nCROSS-VALIDATION RESULTS:")
    for model, metrics in cv_results.items():
        print(f"{model.replace('_', ' ').title()}:")
        print(f"  Brier Score: {metrics['brier_score_mean']:.4f} (±{metrics['brier_score_std']:.4f})")
        print(f"  Expected Profit: {metrics['profit_mean']:.0f}원 (±{metrics['profit_std']:.0f}원)")
    
    print("\nEvaluating models on test set...")
    results = predictor.evaluate_models(X_test, y_test)
    
    # Generate comprehensive report
    predictor.generate_deployment_report(X_test, y_test, results, original_data)
    
    # Print model summary
    predictor.print_model_summary()

if __name__ == "__main__":
    main()
