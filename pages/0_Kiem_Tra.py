import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from services.utils.demo import demo_tab
from services.utils.process import process_input
from services.utils.training import train_process


def main():
    st.title('Alphabet Recognition')
    data_process, train, demo, mlflow_p = st.tabs(
        ["Tiền xử lý dữ liệu", "Huấn Luyện", "Dự Đoán", "MLflow"])

    # --------------- Data Processing ---------------
    with data_process:
        data_X, data_y = process_input()
        if data_X is not None and data_y is not None:
            st.session_state['data_X'] = data_X
            st.session_state['data_y'] = data_y

    # --------------- Training ---------------
    with train:
        train_process()

    # --------------- DEMO Alphabet ---------------
    with demo:
        demo_tab()

    # --------------- MLflow Tracking ---------------
    with mlflow_p:
        pass


if __name__ == '__main__':
    main()
