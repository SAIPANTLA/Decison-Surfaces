import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles,make_blobs,make_moons


# Function to generate U-shaped dataset
def generate_u_shape(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples, 2)
    y = (X[:, 1] > np.abs(X[:, 0] - 0.5)).astype(int)
    return X, y

# Function to generate Two Spirals dataset
def generate_two_spirals(n_points, noise=0.5):
    np.random.seed(0)
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return X, y

# Function to generate XOR dataset
def generate_xor(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples, 2) * 2 - 1
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

# Function to generate Overlap dataset
def generate_overlap(n_samples):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y

#Inno Logo
st.image(r"innomaticslogo.png")

# Set the title of the app
st.title("Decision Surfaces Visualization")

# Create a sidebar
st.sidebar.header("Modifications")

# Choosing the Algorithms
classifiers = st.sidebar.selectbox(
    "Choose Classifier",
    ("KNeighbors","Logistic Regression","SVC","Random Forest","Decision Tree"))

# Add a select box to the sidebar
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Moons","Blobs","U-Shaped","Circles","XOR","Two Spirals","Overlap")
)

if dataset_name in ("Moons","Circles","Two Spirals","Overlap","XOR"):
    noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, 0.1)
if dataset_name == "Circles":
    factor = st.sidebar.slider("Factor", 0.1, 0.99, 0.5, 0.1)
n_samples = 500

# Add sliders to the sidebar based on the dataset
if dataset_name == "Moons":
    X, y = make_moons(n_samples,noise=noise)

elif dataset_name == "Blobs":
    X, y = make_blobs(n_samples,cluster_std=st.sidebar.slider("Cluster Std",0.0,5.0,1.0,0.1))

elif dataset_name == "U-Shaped":
    X, y = generate_u_shape(n_samples)

elif dataset_name == "Circles":
    from sklearn.datasets import make_circles
    X, y = make_circles(n_samples, noise=noise, factor=factor)

elif dataset_name == "XOR":
    X, y = generate_xor(n_samples)

elif dataset_name == "Two Spirals":
    X, y = generate_two_spirals(n_samples,noise=noise)

elif dataset_name == "Overlap":
    X, y = generate_overlap(n_samples)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=st.sidebar.number_input("Random State Number",min_value=0,max_value=1000,step=1))

# Train a classifier
if classifiers=="KNeighbors":
    Metrics=st.sidebar.selectbox("Distance Metric", ("euclidean", "manhattan", "minkowski"))
    Weights=st.sidebar.selectbox("Weights",('uniform', 'distance'))
    Algorithm=st.sidebar.selectbox("Algorithm",('auto', 'ball_tree', 'kd_tree', 'brute'))
    classifier=KNeighborsClassifier(n_neighbors=st.sidebar.number_input("Enter K value",min_value=1,max_value=10),metric=Metrics,weights=Weights,algorithm=Algorithm)
    classifier.fit(X_train, y_train)
elif classifiers=="Logistic Regression":
    multi_class = st.sidebar.selectbox("Multi-class", ("auto", "ovr", "multinomial"))
    max_iter = st.sidebar.slider("Max Iterations", 50, 500, 100,50)
    classifier=LogisticRegression(multi_class=multi_class,max_iter=max_iter)
    classifier.fit(X_train, y_train)
elif classifiers=="SVC":
    kernal=st.sidebar.selectbox("kernal",('linear','poly','rbf','sigmoid'))
    classifier=SVC(kernel=kernal)
    classifier.fit(X_train, y_train)
elif classifiers == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 100, 50)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    classifier=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth) 
    classifier.fit(X_train, y_train)
    
elif classifiers== "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    classifier=DecisionTreeClassifier(max_depth=max_depth)
    classifier.fit(X_train, y_train)
    

# Make predictions
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Plot the decision surface
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(6,3))
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"{dataset_name} Dataset\nAccuracy: {accuracy:.2f}")

# Display the plot in Streamlit
st.pyplot(plt)
