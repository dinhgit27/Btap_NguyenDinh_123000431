# evaluate_system.py
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score
from sklearn.calibration import calibration_curve
from model_utils import DogCatClassifier

# ==========================================
# 1. ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY)
# ==========================================
def calculate_accuracy(classifier, test_dir):
    y_true = []
    y_pred = []

    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                pred_class, _, _ = classifier.predict_from_file(img_path)
                y_true.append(class_name)
                y_pred.append(pred_class)

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return acc

# ==========================================
# 2. PRECISION, RECALL, F1-SCORE
# ==========================================
def calculate_precision_recall_f1(classifier, test_dir):
    y_true = []
    y_pred = []

    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                pred_class, _, _ = classifier.predict_from_file(img_path)
                y_true.append(class_name)
                y_pred.append(pred_class)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=['dog', 'cat'], average=None
    )

    for i, class_name in enumerate(['dog', 'cat']):
        print(f"Class {class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-score:  {f1[i]:.4f}")

    print("Macro averages:")
    print(f"  Precision: {precision.mean():.4f}")
    print(f"  Recall:    {recall.mean():.4f}")
    print(f"  F1-score:  {f1.mean():.4f}")

    return precision, recall, f1

# ==========================================
# 3. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# ==========================================
def plot_confusion_matrix(classifier, test_dir, save_path=None):
    y_true = []
    y_pred = []

    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                pred_class, _, _ = classifier.predict_from_file(img_path)
                y_true.append(class_name)
                y_pred.append(pred_class)

    cm = confusion_matrix(y_true, y_pred, labels=['dog', 'cat'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['dog', 'cat'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()
    return cm

# ==========================================
# 4. ĐƯỜNG CONG ROC VÀ AUC
# ==========================================
def plot_roc_auc(classifier, test_dir):
    y_true = []          
    y_scores = []        

    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                pred_class, _, probs = classifier.predict_from_file(img_path)
                prob_dog = probs.get('dog', 0.0)   
                y_scores.append(prob_dog)
                y_true.append(1 if class_name == 'dog' else 0)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print(f"AUC: {roc_auc:.4f}")
    return roc_auc, fpr, tpr, thresholds

# ==========================================
# 5. THỜI GIAN XỬ LÝ TRUNG BÌNH (INFERENCE TIME)
# ==========================================
def measure_inference_time(classifier, image_path, num_runs=10):
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        classifier.predict_from_file(image_path)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / num_runs
    print(f"Average inference time over {num_runs} runs: {avg_time*1000:.2f} ms")
    return avg_time

# ==========================================
# 6. KÍCH THƯỚC MÔ HÌNH (MODEL SIZE)
# ==========================================
def get_model_size(model_path='model/model.tflite'):
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    print(f"Model size: {size_kb:.2f} KB ({size_mb:.2f} MB)")
    return size_bytes

# ==========================================
# 7. ĐỘ TIN CẬY CỦA XÁC SUẤT (CALIBRATION)
# ==========================================
def plot_calibration_curve(classifier, test_dir, n_bins=10):
    y_true = []
    y_scores = []

    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                pred_class, confidence, probs = classifier.predict_from_file(img_path)
                y_scores.append(confidence)
                y_true.append(1 if pred_class == class_name else 0)  

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_scores, n_bins=n_bins)

    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Fraction of positives (accuracy)")
    plt.title("Calibration plot")
    plt.legend()
    plt.show()

# ==========================================
# 8. TỐI ƯU NGƯỠNG QUYẾT ĐỊNH (THRESHOLD TUNING)
# ==========================================
def find_optimal_threshold(classifier, test_dir):
    y_true = []
    y_scores_dog = [] 

    for class_name in ['dog', 'cat']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                _, _, probs = classifier.predict_from_file(img_path)
                prob_dog = probs.get('dog', 0.0)
                y_scores_dog.append(prob_dog)
                y_true.append(1 if class_name == 'dog' else 0)

    thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (np.array(y_scores_dog) >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Optimal threshold: {best_thresh:.3f} with F1 = {best_f1:.4f}")
    return best_thresh

# ==========================================
# 9. HÀM MAIN - CHẠY TỔNG HỢP ĐÁNH GIÁ
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate Dog/Cat Classifier')
    parser.add_argument('--test_dir', required=True, help='Path to test directory with dog/ and cat/ subfolders')
    parser.add_argument('--model', default='model/model.tflite', help='Path to model file')
    parser.add_argument('--labels', default='model/labels.txt', help='Path to labels file')
    args = parser.parse_args()

    classifier = DogCatClassifier(model_path=args.model, labels_path=args.labels)

    print("\n=== 1. Accuracy ===")
    calculate_accuracy(classifier, args.test_dir)

    print("\n=== 2. Precision/Recall/F1 ===")
    calculate_precision_recall_f1(classifier, args.test_dir)

    print("\n=== 3. Confusion Matrix ===")
    # Sẽ hiển thị một cửa sổ biểu đồ. Bạn cần tắt cửa sổ đó thì code mới chạy tiếp phần dưới
    plot_confusion_matrix(classifier, args.test_dir, save_path='confusion.png')

    print("\n=== 4. ROC AUC ===")
    # Cửa sổ biểu đồ thứ hai
    plot_roc_auc(classifier, args.test_dir)

    print("\n=== 5. Inference Time ===")
    # Tìm 1 ảnh bất kỳ trong thư mục dog để test tốc độ
    sample_img_dir = os.path.join(args.test_dir, 'dog')
    if os.path.exists(sample_img_dir) and len(os.listdir(sample_img_dir)) > 0:
        sample_img = os.path.join(sample_img_dir, os.listdir(sample_img_dir)[0])
        measure_inference_time(classifier, sample_img)
    else:
        print("Không tìm thấy ảnh mẫu trong thư mục dog/ để test tốc độ.")

    print("\n=== 6. Model Size ===")
    get_model_size(args.model)

    print("\n=== 7. Calibration ===")
    # Cửa sổ biểu đồ thứ ba
    plot_calibration_curve(classifier, args.test_dir)

    print("\n=== 8. Threshold Tuning ===")
    find_optimal_threshold(classifier, args.test_dir)

if __name__ == "__main__":
    main()