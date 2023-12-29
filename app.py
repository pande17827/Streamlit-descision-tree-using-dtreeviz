import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from os import system
from graphviz import Source
import dtreeviz
import base64
import logging

logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
# Load the Iris dataset from scikit-learn
iris = load_iris()
X = iris.data[:, :2]  # Using only two features for visualization purposes
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
st.sidebar.markdown("# Decision Tree Classifier")
criterion = st.sidebar.selectbox(
   'Criterion',
   ('gini', 'entropy')
)
splitter = st.sidebar.selectbox(
   'Splitter',
   ('best', 'random')
)
max_depth = int(st.sidebar.number_input('Max Depth'))
min_samples_split = st.sidebar.slider('Min Samples Split', 1, X_train.shape[0], 2,key=1234)

min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, X_train.shape[0], 1,key=1235)

max_features = st.sidebar.slider('Max Features', 1, 2, 2,key=1236)

max_leaf_nodes = int(st.sidebar.number_input('Max Leaf Nodes'))

min_impurity_decrease = st.sidebar.number_input('Min Impurity Decrease')

# Rest of your sidebar inputs...
# Load initial graph
fig, ax = plt.subplots()
# Plot initial graph
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
orig = st.pyplot(fig)


if st.sidebar.button('Run Algorithm'):
    orig.empty()
    if max_depth == 0:
        max_depth = None

    if max_leaf_nodes == 0:
        max_leaf_nodes = None
   
       
    clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, random_state=42,
                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=min_impurity_decrease)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    st.subheader("Accuracy for Decision Tree: " + str(round(accuracy_score(y_test, y_pred), 2)))
    viz_model = dtreeviz.model(clf,
                           X_train=X, y_train=y,
                           feature_names=iris.feature_names,
                           target_name='iris',
                           class_names=iris.target_names)

    v = viz_model.view(scale=1)     # render as SVG into internal object 
    # v.show()                 # pop up window
    # v.save("/tmp/iris.svg")
    def svg_write(svg, center=True):
       """
       Disable center to left-margin align like other objects.
       """
       # Encode as base 64
       b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
   
       # Add some CSS on top
       css_justify = "center" if center else "left"
       css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
       html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'
   
       # Write the HTML
       st.write(html, unsafe_allow_html=True)
       

    
    svg=v.svg()
    svg_write(svg)
    
