🧑‍💻 Face Recognition with PCA (Eigenfaces)
📖 Overview

This project implements a Face Recognition System using Principal Component Analysis (PCA), also known as the Eigenfaces method.
PCA is used to reduce the dimensionality of face images while preserving the most important features. By projecting faces into a lower-dimensional "eigenface space," the system can perform recognition by comparing feature vectors instead of raw pixels.

This technique was one of the earliest breakthroughs in face recognition research, and while deep learning now dominates the field, eigenfaces are still a powerful way to understand dimensionality reduction, pattern recognition, and linear algebra in computer vision.

⚡ Features

📂 Load and preprocess grayscale face images.

🧮 Apply PCA to compute eigenfaces.

👤 Project images into eigenface space.

🔍 Classify using Euclidean distance and Cosine similarity.

📊 Visualize dataset samples, mean face, and top eigenfaces.

🖼️ Display recognition results (query image vs best match).

🛠️ Tech Stack

Python 3

NumPy
 – linear algebra & matrix operations

OpenCV
 – image processing

Matplotlib
 – visualization

Scikit-learn (optional)
 – for PCA comparison

🗂️ Dataset

The project uses the ORL Face Database (also known as the AT&T face dataset).

Contains 40 individuals, with 10 images per person.

Images are grayscale (.pgm format), size 92×112 pixels.

Variations include facial expressions, lighting, and pose.

📦 In this notebook, the dataset is provided as a Faces.zip file containing all .pgm images.
If you don’t have it, you can download the ORL Faces dataset from public sources.

🚀 How It Works

Data Preparation

Load all .pgm images from the dataset.

Flatten each image into a 1D vector.

Split into training and testing sets.

Mean Face

Compute the average face across the dataset.

Subtract the mean face from all images → mean-centered data.

PCA & Eigenfaces

Compute the covariance matrix.

Perform eigenvalue decomposition.

Sort eigenvectors by largest eigenvalues → eigenfaces.

Keep top k eigenfaces (e.g., 50) for dimensionality reduction.

Projection into Eigenface Space

Represent each face as a weighted sum of eigenfaces.

Store the weight vectors as face signatures.

Recognition

For a test face, compute its projection (signature).

Compare with training signatures using:

Euclidean distance

Cosine similarity

Return the closest match.

Visualization

Display sample faces, mean face, and eigenfaces.

Show query vs predicted match side by side.

📊 Results

The first 16 eigenfaces are visualized—these capture variations in lighting, facial structure, and shadows.

Recognition achieves good results on clean images with low noise.

Both Euclidean distance and Cosine similarity perform well for matching.

📷 Example output (add screenshots from your notebook):

Mean face

Top eigenfaces

Query vs best match

🔮 Future Improvements

✅ Increase number of eigenfaces for higher accuracy.

✅ Use k-NN classifier instead of direct similarity measures.

✅ Add noise robustness and face preprocessing.

✅ Extend to larger datasets (e.g., Yale, LFW).

✅ Compare performance with deep learning models (CNNs).

📌 Learning Objectives

This project helped me practice:

Dimensionality reduction with PCA.

Matrix algebra in computer vision.

Building a basic face recognition pipeline.

Visualizing high-dimensional data.

Evaluating classification performance.

▶️ Running the Project

Clone the repository:

git clone https://github.com/yourusername/eigenfaces-face-recognition.git
cd eigenfaces-face-recognition


Install dependencies:

pip install numpy opencv-python matplotlib


Run the Jupyter Notebook:

jupyter notebook Linear_project_sp25.ipynb


Make sure the Faces.zip dataset is in the same directory.

🙌 Acknowledgments

AT&T Laboratories Cambridge – ORL Face Dataset.

Eigenfaces method by Turk and Pentland (1991).
