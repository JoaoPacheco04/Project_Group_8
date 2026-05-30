🌸 Anime Data Analytics & Machine Learning Hub
An end-to-end Data Science and Machine Learning platform designed to analyze, predict, and recommend animes. Developed as the final practical work for the Data Analysis Lab (Degree in Information Technology), this project features a robust analytical pipeline, diverse predictive models, and an interactive frontend built with Streamlit.

🌟 Project Overview
This platform allows users to explore a massive Anime dataset through dynamic dashboards and powerful Machine Learning integrations. It covers the entire data science lifecycle: from exploratory data analysis (EDA) and feature engineering to complex model training, clustering, and deploying a recommendation engine—all accessible via a user-friendly graphical interface.

🚀 Key Features
📊 Exploratory Data Analysis (EDA): Comprehensive statistical analysis, feature engineering, and dynamic visualizations of anime genres, ratings, and user preferences.

🧠 Comprehensive ML Suite: Implementation and comparison of multiple algorithms including Linear/Logistic Regression, SVM (multi-kernel), K-NN, Decision Trees, Random Forests, Naïve Bayes, and Neural Networks.

🎯 Anime Recommendation Engine: A specialized recommendation algorithm (Challenge feature) tailored to suggest animes based on user preferences and dataset similarities.

🧩 Dimensionality Reduction & Clustering: Advanced SVD-based Principal Component Analysis (PCA) and Hierarchical Clustering with dendrogram visualizations for optimal grouping.

⚙️ Automated Tuning & Tracking: Model hyperparameter optimization using GridSearch and Optuna, with versioning and experiment tracking handled by MLflow.

🖥️ Interactive Dashboard: A dynamic Streamlit GUI that allows users to filter data, visualize dataset distributions, and test the best predictive models in real-time.

🛠️ Technical Stack
Data Science & Machine Learning (The Core)

Python 3.10+ (Core programming language)

Pandas & NumPy (Data manipulation and linear algebra)

Scikit-Learn (Machine Learning, Preprocessing, and Clustering)

Optuna & MLflow (Hyperparameter optimization and model version tracking)

SciPy (Statistical functions and Hierarchical Clustering/Dendrograms)

Visualization & Frontend (The Experience)

Streamlit (Interactive Dashboard GUI)

Plotly & Altair (Dynamic, interactive charts for the dashboard)

Matplotlib & Seaborn (Static statistical plotting and correlation matrices)

🏗️ Architecture & Best Practices
Robust Data Preprocessing: Strict implementation of data normalization, standardization, and missing value handling prior to model training.

Model Evaluation Pipeline: Extensive use of Cross-Validation to ensure model reliability, evaluated through RMSE, Precision, Recall, and F1-measure metrics.

Separation of Concerns: Clean codebase dividing data processing scripts, model training pipelines, and the Streamlit frontend application.

Reproducibility: Strict tracking of fit times, accuracy comparisons, and algorithm states to easily compare pre-PCA and post-PCA performance.

🔧 How to Run
1. Environment Setup

Clone the repository and navigate to the project folder.

Create a virtual environment: python -m venv venv

Activate the environment:

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

Install dependencies: pip install -r requirements.txt

2. Model Tracking (MLflow)

To view the model comparisons, metrics, and versions, start the MLflow UI:

Bash
mlflow ui
Access the MLflow dashboard at http://localhost:5000.

3. Launch the Dashboard (Streamlit)

Start the interactive dashboard by running:

Bash
streamlit run app.py
The application will automatically open in your browser at http://localhost:8501.
