import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def process_input():
    st.header("Tiền xử lý dữ liệu")
    
    # Option to use sample data
    use_sample = st.checkbox("Sử dụng dữ liệu mẫu có sẵn")
    
    data_X = None
    data_y = None
    
    if use_sample:
        # Load NumPy data files
        try:
            data_X = np.load('services/data/alphabet_X.npy')
            data_y = np.load('services/data/alphabet_y.npy')
            st.success(f"Đã tải dữ liệu mẫu")
            
            # Display data info
            st.subheader("Thông tin dữ liệu")
            
            # Display data shape
            st.write(f"Kích thước dữ liệu X: {data_X.shape}")
            st.write(f"Kích thước dữ liệu y: {data_y.shape}")
            
            # Display data types
            st.write(f"Kiểu dữ liệu X: {data_X.dtype}")
            st.write(f"Kiểu dữ liệu y: {data_y.dtype}")
            
            # Display sample images if data_X contains image data
            if len(data_X.shape) >= 3:  # Assuming data_X contains images
                st.subheader("Mẫu hình ảnh")
                
                # Get unique classes
                unique_classes = np.unique(data_y)
                
                # Display one sample from each of the first 10 classes
                num_classes_to_show = min(10, len(unique_classes))
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                axes = axes.flatten()
                
                for i in range(num_classes_to_show):
                    # Find first index of this class
                    class_idx = np.where(data_y == unique_classes[i])[0][0]
                    
                    if len(data_X.shape) == 4:  # Color images (samples, height, width, channels)
                        img = data_X[class_idx]
                    else:  # Grayscale images (samples, height, width)
                        img = data_X[class_idx].reshape(data_X.shape[1], data_X.shape[2])
                    
                    axes[i].imshow(img, cmap='gray')
                    axes[i].set_title(f"Class: {unique_classes[i]}")
                    axes[i].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
                
                # Display class distribution
                st.subheader("Phân phối các lớp")
                unique_classes, class_counts = np.unique(data_y, return_counts=True)
                
                # Create DataFrame for better visualization
                class_df = pd.DataFrame({
                    'Lớp': unique_classes,
                    'Số lượng': class_counts
                })
                
                st.write(class_df)
                
                # Plot class distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(unique_classes, class_counts)
                ax.set_xlabel('Lớp')
                ax.set_ylabel('Số lượng')
                ax.set_title('Phân phối các lớp')
                st.pyplot(fig)
            
            # Store data in session state
            st.session_state['data_X'] = data_X
            st.session_state['data_y'] = data_y
            
            return data_X, data_y
            
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {e}")
    else:
        # File uploader for NumPy files
        st.subheader("Tải lên file dữ liệu")
        uploaded_X = st.file_uploader("Tải lên file X (features)", type=["npy"])
        uploaded_y = st.file_uploader("Tải lên file y (labels)", type=["npy"])
        
        if uploaded_X is not None and uploaded_y is not None:
            try:
                data_X = np.load(uploaded_X)
                data_y = np.load(uploaded_y)
                
                st.success(f"Đã tải lên file X: {uploaded_X.name} và file y: {uploaded_y.name}")
                
                # Display data info
                st.subheader("Thông tin dữ liệu")
                
                # Display data shape
                st.write(f"Kích thước dữ liệu X: {data_X.shape}")
                st.write(f"Kích thước dữ liệu y: {data_y.shape}")
                
                # Store data in session state
                st.session_state['data_X'] = data_X
                st.session_state['data_y'] = data_y
                
                return data_X, data_y
                
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {e}")
    
    return None, None

