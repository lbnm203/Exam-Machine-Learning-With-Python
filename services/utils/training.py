import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import os
import time
import datetime

def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Exam-Machine-Learning-With-Python.mlflow/"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

    mlflow.set_experiment("KTHP_Machine_Learning")

def train_process():
    st.header("Huấn luyện mô hình")
    st.write("Chọn số lượng ảnh, tỷ lệ phân chia dữ liệu, số fold cho Cross Validation và mô hình cần huấn luyện.")
    
    # Kiểm tra dữ liệu có trong session_state không
    if "data_X" not in st.session_state or "data_y" not in st.session_state:
        st.warning("Chưa có dữ liệu được tải lên. Vui lòng chuyển sang tab 'Tiền xử lý dữ liệu' để tải dữ liệu và quay lại đây.")
        return

    X = st.session_state["data_X"]
    y = st.session_state["data_y"]
    total_samples = X.shape[0]
    
    st.write(f"Tổng số ảnh: {total_samples}")

    # Chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:",
                            min_value=1000, max_value=total_samples, value=min(10000, total_samples),
                            step=1000, help="Chọn số lượng ảnh con để sử dụng trong quá trình huấn luyện")
    if num_samples >= total_samples:
        num_samples = total_samples - 10  # Để lại ít nhất 1 mẫu cho tập test
    st.session_state.total_samples = num_samples

    # Chọn tỷ lệ phân chia dữ liệu
    test_size = st.slider("Chọn % dữ liệu Test", min_value=10, max_value=50, value=20,
                          help="Chọn phần trăm dữ liệu cho tập Test")
    val_size = st.slider("Chọn % dữ liệu Validation", min_value=0, max_value=50, value=10,
                         help="Chọn phần trăm dữ liệu cho tập Validation")
    remaining_size = 100 - test_size

    # Chọn số Fold cho Cross Validation
    k_folds = st.slider("Chọn số Fold cho Cross Validation:", min_value=2, max_value=10, value=5,
                        help="Chọn số fold sử dụng trong cross validation")

    # Option chọn mô hình cần huấn luyện
    model_option = st.selectbox("Chọn mô hình cần huấn luyện:", 
                                options=["Logistic Regression", "KNN"],
                                help="Chọn mô hình muốn sử dụng cho quá trình huấn luyện")

    # Tùy chọn lưu mô hình vào MLflow
    use_mlflow = st.checkbox("Lưu mô hình vào MLflow", value=True, 
                            help="Lưu thông tin mô hình vào MLflow để theo dõi và so sánh")
    
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs("services/models", exist_ok=True)

    if st.button("Bắt đầu huấn luyện"):
        st.write("**Bắt đầu chọn số lượng ảnh và chia dữ liệu...**")
        # Chọn subset của dữ liệu theo số lượng ảnh mong muốn
        X_selected, _, y_selected, _ = train_test_split(
            X, y, train_size=num_samples, stratify=y, random_state=42
        )
        
        # Chia train/test theo tỷ lệ đã chọn
        stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
        )
        # Chia train/validation theo tỷ lệ đã chọn trên phần còn lại
        stratify_option = y_train_full if len(np.unique(y_train_full)) > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size/remaining_size,
            stratify=stratify_option, random_state=42
        )

        # Nếu dữ liệu ảnh có nhiều chiều (ví dụ 3 hay 4 chiều), flatten dữ liệu để mô hình có thể xử lý
        if X_train.ndim > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        st.write("**Kích thước các tập dữ liệu sau khi chia:**")
        st.write(f"- Train: {X_train.shape[0]} mẫu")
        st.write(f"- Validation: {X_val.shape[0]} mẫu")
        st.write(f"- Test: {X_test.shape[0]} mẫu")
        
        # Khai báo mô hình dựa trên lựa chọn
        models = {}
        if model_option == "Logistic Regression":
            models["Logistic Regression"] = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto', random_state=42)
        elif model_option == "KNN":
            models["KNN"] = KNeighborsClassifier()

        if not models:
            st.error("Chưa có mô hình nào được chọn để huấn luyện.")
            return

        # Tạo progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Huấn luyện và đánh giá từng mô hình
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Đang huấn luyện mô hình {name}...")
            progress_bar.progress((i * 33) % 100)
            
            try:
                # Cross Validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=k_folds)
                st.write(f"**{name}:**")
                
                # Hiển thị accuracy của từng fold
                st.write("**Accuracy của từng fold:**")
                for fold_idx, score in enumerate(cv_scores, 1):
                    st.write(f"- Fold {fold_idx}: {score:.4f}")
                
                # Hiển thị accuracy trung bình
                mean_cv = cv_scores.mean()
                std_cv = cv_scores.std()
                st.write(f"**Accuracy trung bình:** {mean_cv:.4f} (±{std_cv:.4f})")
                
                # Huấn luyện mô hình
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Đánh giá trên tập validation
                val_acc = model.score(X_val, y_val)
                st.write(f"**{name} Accuracy trên Validation:** {val_acc:.4f}")
                
                # Đánh giá trên tập test
                test_acc = model.score(X_test, y_test)
                st.write(f"**{name} Accuracy trên Test:** {test_acc:.4f}")
                
                # Lưu mô hình vào MLflow nếu được chọn
                if use_mlflow:
                    # Tạo tên run dựa trên thời gian hiện tại
                    run_name = f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    with mlflow.start_run(run_name=run_name):
                        # Log parameters
                        mlflow.log_param("model_type", name)
                        mlflow.log_param("num_samples", num_samples)
                        mlflow.log_param("test_size", test_size)
                        mlflow.log_param("val_size", val_size)
                        mlflow.log_param("k_folds", k_folds)
                        
                        # Log metrics
                        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
                        mlflow.log_metric("validation_accuracy", val_acc)
                        mlflow.log_metric("test_accuracy", test_acc)
                        mlflow.log_metric("training_time", training_time)
                        
                        # Log model
                        mlflow.sklearn.log_model(model, "model")
                        
                        # Lấy run ID để hiển thị
                        run_id = mlflow.active_run().info.run_id
                        st.success(f"Đã lưu mô hình {name} vào MLflow với Run ID: {run_id}")
                
                # Lưu mô hình vào file
                import joblib
                model_path = f"services/models/{name.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                joblib.dump(model, model_path)
                st.success(f"Đã lưu mô hình {name} vào file: {model_path}")
                
                # Cập nhật progress bar
                progress_bar.progress((i * 33 + 33) % 100)
                
            except Exception as e:
                st.error(f"Lỗi khi huấn luyện với {name}: {e}")
        
        # Hoàn thành progress bar
        progress_bar.progress(100)
        status_text.text("Hoàn thành huấn luyện!")
        
        # Lưu mô hình vào session_state
        st.session_state["trained_models"] = models
        st.success("Đã lưu các mô hình vào session_state để sử dụng trong tab Demo")

