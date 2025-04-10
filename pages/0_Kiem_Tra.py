import streamlit as st
import pandas as pd

from services.utils.training import mlflow_input
from services.utils.show_mlflow import show_experiment_selector


def main():
    st.title('Machine Learning With Python')
    data_pages, theory, train, demo, mlflow_p = st.tabs(
        ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "Demo", "MLflow Tracking"])

    # --------------- Data MNIST ---------------
    with data_pages:
        # X, y = mnist_dataset()
        pass

    # -------- Theory Decision Tree - SVM ---------
    with theory:
        # neural_network()
        pass

    # --------------- Training ---------------
    with train:
        # train_process(X, y)
        mlflow_input()
        # pass

    # --------------- DEMO MNIST ---------------
    with demo:
        # demo_app()
        pass

    # --------------- MLflow Tracking ---------------
    with mlflow_p:
        show_experiment_selector()


if __name__ == '__main__':
    main()
