import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve,precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score, balanced_accuracy_score



# Header and text
st.title("Diabetes prediction dataset DASHBOARD")
st.write("""This dashboard will present the info about the Diabetes prediction dataset from Kaggle (https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).""")

# Настройка боковой панели
st.sidebar.title("About")
st.sidebar.info(
    """
    This app is Training dashboard.
    """
)
st.sidebar.info("Feel free to contact me "
                "[here](https://github.com/PeterOstr).")


page = st.sidebar.selectbox("Choose page",
                            ["Charts",
                             "Other"])

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload y_pred.csv file", type=["csv"])

# Placeholder for dataframes
y = pd.DataFrame()
y_pred = pd.DataFrame()

button_answer = st.sidebar.button('Check')


if (uploaded_file is not None) and button_answer:

    y_pred = pd.read_csv(uploaded_file)
    y = pd.read_csv('y.csv')

    if 'diabetes' in y_pred.columns:
        X = y_pred.drop('diabetes',axis=1)
        y = y_pred['diabetes']

    else:
        X = y_pred
        st.write('file without target')

    X_dict = X.reset_index().to_dict(orient='list')
    response_prediction = requests.post('https://first-try-6o4cxe37ma-uc.a.run.app/model/predict',  json=X_dict)
    resonse_results = pd.read_json(response_prediction.json()).set_index('index')
    st.table(resonse_results.head())

if (uploaded_file is None) and (button_answer):
    st.write('file not uploaded')



#
# if page == "Charts":
#     st.header("""Charts Demo""")
#
#     if not y_pred.empty:
#         st.write('f1_score:', np.round(f1_score(y, y_pred), 3))
#         st.write('r2_score:', np.round(r2_score(y_pred, y), 3))
#         st.write('balanced_accuracy_score:', np.round(balanced_accuracy_score(y_pred, y), 3))
#
#         # Calculate ROC AUC
#         fpr, tpr, thresholds = roc_curve(y, y_pred)
#         roc_auc = roc_auc_score(y, y_pred)
#
#         # Plot ROC curve
#         plt.figure(figsize=(8, 8))
#         plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic (ROC) Curve')
#         plt.legend(loc='lower right')
#         st.pyplot(plt)
#
#         # Plot precision recall
#         # calculate precision and recall
#         precision, recall, thresholds = precision_recall_curve(y_pred, y)
#
#         # create precision recall curve
#         fig, ax = plt.subplots()
#         ax.plot(recall, precision, color='purple')
#
#         # add axis labels to plot
#         ax.set_title('Precision-Recall Curve')
#         ax.set_ylabel('Precision')
#         ax.set_xlabel('Recall')
#
#         # display plot
#         st.pyplot(plt)
#
#         # Confusion Matrix
#         cm = confusion_matrix(y, y_pred)
#
#         # Normalize the confusion matrix
#         cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#         # Display the confusion matrix
#         fig, ax = plt.subplots()
#         ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
#         ax.set_title('Confusion Matrix (Normalized)')
#         st.pyplot(fig)
#
#     y_pred = pd.read_csv('y_pred.csv')
#     y = pd.read_csv('y.csv')
#
#     st.write('f1_score:',np.round(f1_score(y, y_pred),3))
#     st.write('r2_score:',np.round(r2_score(y_pred, y),3))
#     st.write('balanced_accuracy_score:',np.round(balanced_accuracy_score(y_pred,y),3))
#
#
#     # Calculate ROC AUC
#     fpr, tpr, thresholds = roc_curve(y, y_pred)
#     roc_auc = roc_auc_score(y, y_pred)
#
#     # Plot ROC curve
#     plt.figure(figsize=(8, 8))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc='lower right')
#     st.pyplot(plt)
#
#     # Plot precision recall
#     #calculate precision and recall
#     precision, recall, thresholds = precision_recall_curve(y_pred, y)
#
#     #create precision recall curve
#     fig, ax = plt.subplots()
#     ax.plot(recall, precision, color='purple')
#
#     #add axis labels to plot
#     ax.set_title('Precision-Recall Curve')
#     ax.set_ylabel('Precision')
#     ax.set_xlabel('Recall')
#
#     #display plot
#     st.pyplot(plt)
#
#
#     # Confusion Matrix
#     cm = confusion_matrix(y, y_pred)
#
#     # Normalize the confusion matrix
#     cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     # Display the confusion matrix
#     fig, ax = plt.subplots()
#     ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=[0, 1]).plot(cmap='Blues', ax=ax)
#     ax.set_title('Confusion Matrix (Normalized)')
#     st.pyplot(fig)
#
#
#
# elif page == "Other ":
#     st.header("""Other test page:""")



