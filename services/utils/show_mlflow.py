from datetime import datetime
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import logging
import numpy as np  # Thêm import numpy


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
        model = run.data.params.get("model", "N/A")
        data.append({
            "Tên Run": run_name,
            "Run ID": run.info.run_id
        })

    df = pd.DataFrame(data, dtype='object')
    # st.dataframe(df, hide_index=True, width=1200)

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

# Giao diện Streamlit (chỉ hiển thị log huấn luyện)


def show_experiment_selector():
    st.title("MLFflow Tracking")

    # Tạo client MLflow
    client = MlflowClient()

    # Chỉ hiển thị log từ MNIST_Dimensionality_Reduction (huấn luyện)
    with st.spinner("Đang tải log huấn luyện..."):
        train_df, train_runs = display_logs(
            client, "KTHP_Machine_Learning")
        # Thêm nút làm mới cache với key duy nhất
    if st.button("🔄 Làm mới dữ liệu", key=f"refresh_data_{datetime.now().microsecond}"):
        st.cache_data.clear()
        st.rerun()

    experiment = client.get_experiment_by_name("KTHP_Machine_Learning")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    run_names_show = [run.data.tags.get(
        "mlflow.runName", run.info.run_id) for run in runs]
    selected_run_name_show = st.selectbox(
        "Hiển thị Runs", run_names_show)
    selected_run_id_show = next(run.info.run_id for run in runs if run.data.tags.get(
        "mlflow.runName", run.info.run_id) == selected_run_name_show)
    if selected_run_name_show:
        selected_run = client.get_run(selected_run_id_show)
        st.subheader(f"📌 Thông tin Run: {selected_run_name_show}")
        st.write(f"**Run ID:** {selected_run_id_show}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")

        # Thời gian lưu dưới dạng milliseconds
        start_time_ms = selected_run.info.start_time
        if start_time_ms:
            start_time = datetime.fromtimestamp(
                start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"

        st.write(f"**Thời gian chạy:** {start_time}")
        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

    if train_runs:
        st.write("---")
        st.write("## Xóa Log")
        if train_df is not None and not train_df.empty:
            # Get the list of Run IDs from the training dataframe
            # train_run_ids = train_df["Run ID"].tolist()
            run_names = [run.data.tags.get(
                "mlflow.runName", run.info.run_id) for run in train_runs]
            # Allow users to select runs to delete
            selected_train_runs = st.multiselect(
                "Chọn runs để xóa", run_names)
            if st.button("Xóa runs đã chọn", key="delete_train_runs"):
                clear_selected_logs(client, selected_train_runs)

        st.write("---")
        st.write("## Cập nhật tên Run")
        # Cập nhật tên run
        experiment = client.get_experiment_by_name("MNIST_Neural_Network")
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_names = [run.data.tags.get(
            "mlflow.runName", run.info.run_id) for run in runs]
        selected_run_name = st.selectbox(
            "Chọn Run ID để cập nhật tên", run_names)
        selected_run_id = next(run.info.run_id for run in runs if run.data.tags.get(
            "mlflow.runName", run.info.run_id) == selected_run_name)
        new_run_name = st.text_input("Tên Run mới", key="new_run_name")

        if st.button("Cập nhật tên", key="update_run_name"):
            try:
                client.set_tag(selected_run_id, "mlflow.runName", new_run_name)
                st.success(f"Đã cập nhật tên run thành: {new_run_name}")
            except Exception as e:
                st.error(f"Lỗi khi cập nhật tên: {e}")