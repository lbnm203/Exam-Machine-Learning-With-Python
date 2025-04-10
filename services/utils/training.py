import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split,  StratifiedKFold
import os
import mlflow.keras
from tensorflow import keras
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Exam-Machine-Learning-With-Python.mlflow/"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

    mlflow.set_experiment("KTHP_Machine_Learning")
