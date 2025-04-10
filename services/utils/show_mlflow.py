from datetime import datetime
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import logging
import numpy as np

# Hàm hiển thị và xóa log từ experiment
@st.cache_data
def display_logs(_client, experiment_name):
    experiment = _client.get_experiment_by_name(experiment_name)
    if not experiment:
        st.warning(
            f"Chưa có experiment '{experiment_name}'. Sẽ tạo khi có log đầu tiên.")
        return None, None

    runs = _client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        st.warning(f"Không có log nào trong experiment '{experiment_name}'.")
        return None, None

    data = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        model = run.data.params.get("model_type", "N/A")
        accuracy = run.data.metrics.get("test_accuracy", 0)
        start_time = datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M:%S")
        
        data.append({
            "Tên Run": run_name,
            "Run ID": run.info.run_id,
            "Mô hình": model,
            "Accuracy": f"{accuracy:.4f}",
            "Thời gian": start_time
        })

    df = pd.DataFrame(data, dtype='object')
    return df, runs

# Hàm xóa log theo lựa chọn
def clear_selected_logs(client, selected_runs):
    if not selected_runs:
        st.warning("Vui lòng chọn ít nhất một run để xóa.")
        return

    with st.spinner("Đang xóa các run đã chọn..."):
        for run_id in selected_runs:
            client.delete_run(run_id)
        st.success(f"Đã xóa {len(selected_runs)} run thành công!")
    st.rerun()

# Giao diện Streamlit cho MLflow Tracking
def show_experiment_selector():
    st.title("MLFlow Tracking")

    # Tạo client MLflow
    client = MlflowClient()

    # Hiển thị log từ experiment
    experiment_name = "KTHP_Machine_Learning"
    with st.spinner("Đang tải log..."):
        logs_df, runs = display_logs(client, experiment_name)
        
    # Thêm nút làm mới cache với key duy nhất
    if st.button("🔄 Làm mới dữ liệu", key=f"refresh_data_{datetime.now().microsecond}"):
        st.cache_data.clear()
        st.rerun()

    # Hiển thị bảng log nếu có dữ liệu
    if logs_df is not None and not logs_df.empty:
        st.subheader("📋 Danh sách các Run")
        st.dataframe(logs_df, hide_index=True)
        
        # Hiển thị chi tiết run được chọn
        st.subheader("📊 Chi tiết Run")
        run_names = [run.data.tags.get("mlflow.runName", run.info.run_id) for run in runs]
        selected_run_name = st.selectbox("Chọn Run để xem chi tiết", run_names)
        
        if selected_run_name:
            selected_run_id = next(run.info.run_id for run in runs if run.data.tags.get(
                "mlflow.runName", run.info.run_id) == selected_run_name)
            selected_run = client.get_run(selected_run_id)
            
            # Hiển thị thông tin cơ bản
            st.write(f"**Run ID:** {selected_run_id}")
            st.write(f"**Trạng thái:** {selected_run.info.status}")
            
            # Thời gian
            start_time_ms = selected_run.info.start_time
            if start_time_ms:
                start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_time = "Không có thông tin"
            st.write(f"**Thời gian chạy:** {start_time}")
            
            # Hiển thị parameters
            params = selected_run.data.params
            if params:
                st.write("### ⚙️ Parameters:")
                params_df = pd.DataFrame({"Giá trị": params.values()}, index=params.keys())
                st.dataframe(params_df)
            
            # Hiển thị metrics
            metrics = selected_run.data.metrics
            if metrics:
                st.write("### 📊 Metrics:")
                metrics_df = pd.DataFrame({"Giá trị": metrics.values()}, index=metrics.keys())
                st.dataframe(metrics_df)
        
        # Phần xóa runs
        st.write("---")
        st.subheader("🗑️ Xóa Runs")
        selected_runs_to_delete = st.multiselect(
            "Chọn runs để xóa", run_names)
        
        if selected_runs_to_delete:
            selected_run_ids = [next(run.info.run_id for run in runs if run.data.tags.get(
                "mlflow.runName", run.info.run_id) == name) for name in selected_runs_to_delete]
            
            if st.button("Xóa runs đã chọn"):
                clear_selected_logs(client, selected_run_ids)
