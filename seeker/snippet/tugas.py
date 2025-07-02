#date: 2025-07-02T17:04:04Z
#url: https://api.github.com/gists/223400de2a127d6aec056f3778c19b61
#owner: https://api.github.com/users/darylkusdinar

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris # Menggunakan dataset iris bawaan scikit-learn

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split-out validation dataset
# Using random_state=1 for reproducibility as in the PDF 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1) # 

# Define models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) # [cite: 1509, 1510]
models.append(('LDA', LinearDiscriminantAnalysis())) # [cite: 1511]
models.append(('KNN', KNeighborsClassifier())) # [cite: 1512]
models.append(('CART', DecisionTreeClassifier())) # [cite: 1513]
models.append(('NB', GaussianNB())) # [cite: 1514]
models.append(('SVM', SVC(gamma='auto'))) # [cite: 1515]

# --- Function to calculate metrics per class (one-vs-rest) ---
def calculate_metrics_one_vs_rest(y_true, y_pred, target_class_label, class_names):
    tp, fn, fp, tn = 0, 0, 0, 0

    # Map target_class_label to its integer index
    target_class_index = list(class_names).index(target_class_label)

    for i in range(len(y_true)):
        is_target_true = (y_true[i] == target_class_index)
        is_target_pred = (y_pred[i] == target_class_index)

        if is_target_true and is_target_pred:
            tp += 1
        elif is_target_true and not is_target_pred:
            fn += 1
        elif not is_target_true and is_target_pred:
            fp += 1
        else: # not is_target_true and not is_target_pred
            tn += 1

    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}


# --- Perform predictions and gather results ---
summary_results = {}

for name, model in models: # [cite: 1519]
    model.fit(X_train, Y_train) # 
    predictions = model.predict(X_validation) # [cite: 1569]

    # Overall accuracy
    overall_accuracy = accuracy_score(Y_validation, predictions) # [cite: 1577]

    # Metrics for each class
    class_metrics = {}
    for i, class_name in enumerate(target_names):
        # Convert predictions and true labels to one-vs-rest format for specific class
        y_true_binary = (Y_validation == i).astype(int)
        y_pred_binary = (predictions == i).astype(int)

        # Using scikit-learn's built-in metrics for simplicity and consistency
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # Also calculate manually for comparison (as per challenge hint)
        manual_metrics = calculate_metrics_one_vs_rest(Y_validation, predictions, class_name, target_names)

        class_metrics[class_name] = {
            'Precision (sklearn)': precision,
            'Recall (sklearn)': recall,
            'F1-Score (sklearn)': f1,
            'Precision (manual)': manual_metrics['Precision'],
            'Recall (manual)': manual_metrics['Recall'],
            'F1-Score (manual)': manual_metrics['F1-Score']
        }
    
    summary_results[name] = {
        'Overall Accuracy': overall_accuracy,
        'Class Metrics': class_metrics
    }

# --- Display Results in a Structured Table ---
print("--- Ringkasan Hasil Prediksi per Algoritma dan Kelas ---")

for algo_name, data in summary_results.items():
    print(f"\nAlgoritma: {algo_name}")
    print(f"Akurasi Keseluruhan: {data['Overall Accuracy']:.4f}")

    # Create a DataFrame for class-specific metrics for better presentation
    df_metrics = pd.DataFrame(data['Class Metrics']).T # Transpose to have classes as rows
    print(df_metrics.to_string())

# --- Perbandingan dengan Output PDF ---
print("\n--- Perbandingan dengan Hasil di PDF (untuk referensi) ---")

# Data dari slide 32 
print("\nMetrik dari Cross-Validation (Slide 32):")
print("LR: 0.941667 (0.065085)") # 
print("LDA: 0.975000 (0.038188)") # 
print("KNN: 0.958333 (0.041667)") # 
print("CART: 0.950000 (0.055277)") # 
print("NB: 0.950000 (0.055277)") # 
print("SVM: 0.983333 (0.033333)") # 

# Data dari tabel evaluasi prediksi SVM (Slide 37) 
print("\nEvaluasi Prediksi SVM (Slide 37):")
print("Accuracy: 0.9666666666666667") # [cite: 1585]
print("Iris-setosa Precision: 1.00, Recall: 1.00, F1-score: 1.00") # 
print("Iris-versicolor Precision: 1.00, Recall: 0.92, F1-score: 0.96") # 
print("Iris-virginica Precision: 0.86, Recall: 1.00, F1-score: 0.92") # 

# Catatan: Akurasi keseluruhan yang saya hitung akan sedikit berbeda dari PDF
# karena saya melatih model sekali pada X_train dan menguji pada X_validation.
# PDF untuk slide 32  menampilkan hasil cross-validation,
# dan untuk slide 37  mungkin menggunakan set validasi yang berbeda
# atau seed acak yang berbeda untuk split data jika tidak disebutkan.
# Akurasi yang dihitung di kode ini adalah untuk satu split train/validation (80/20)
# dengan random_state=1, sesuai dengan cara data dibagi di slide 30.