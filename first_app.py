import pandas as pd
import numpy as np
import streamlit as st
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


if page == "Charts":
    st.header("""Charts Demo""")



    y_pred = pd.read_csv('y_pred.csv')
    y = pd.read_csv('y.csv')

    st.write('f1_score:',np.round(f1_score(y, y_pred),3))
    st.write('r2_score:',np.round(r2_score(y_pred, y),3))
    st.write('balanced_accuracy_score:',np.round(balanced_accuracy_score(y_pred,y),3))


    # Calculate ROC AUC
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    # Plot precision recall
    #calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(y_pred, y)

    #create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')

    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    #display plot
    st.pyplot(plt)


    #Confusion Matrix

    ConfusionMatrixDisplay.from_predictions(y, y_pred)

    st.pyplot(plt)

elif page == "Iris Dataset":
    st.header("""Сгенерировать N случайных событий из распределения Фреше с функцией распределения:""")



