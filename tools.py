import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import shapiro, probplot
from sklearn.neighbors import KernelDensity
from scipy.stats import shapiro


def neg_log_likelihood(params, data):
    mean, std_dev = params
    return -np.sum(norm.logpdf(data, loc=mean, scale=std_dev))

# Shapiro-Wilk Test


def shapiro_wilk(data):
    stat, p_value = shapiro(data)
    print("Shapiro-Wilk Test:")
    print("Test Statistic:", stat)
    print("P-value:", p_value)

    # Interpret the results
    significance_level = 0.05
    if p_value > significance_level:
        print("The data appears to be normally distributed (fail to reject the null hypothesis)")
    else:
        print("The data does not appear to be normally distributed (reject the null hypothesis)")


""" 
Maximum Likelihood Estimation. Assuming that accuracy data is normally distributed, 
find mean and variance that maximize likelihood
"""


def maximum_likelihood_estimation(accuracy_scores):
    # Initial guess for mean and standard deviation
    initial_guess = [np.mean(accuracy_scores), np.std(accuracy_scores)]
    # Maximize the likelihood using scipy's minimize function
    result = minimize(neg_log_likelihood, initial_guess,
                      args=(accuracy_scores,))
    estimated_mean, estimated_std_dev = result.x

    print("Estimated Mean:", estimated_mean)
    print("Estimated Standard Deviation:", estimated_std_dev)

    # test normality
    # Plot histogram
    plt.hist(accuracy_scores, bins=20, density=True, alpha=0.6, color='g')

    # Plot the estimated normal distribution based on MLE
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, estimated_mean, estimated_std_dev)
    plt.plot(x, p, 'k', linewidth=2)

    plt.title("Histogram of Accuracy Scores and Estimated Normal Distribution")
    plt.show()

    # Shapiro-Wilk test
    shapiro_wilk(accuracy_scores)


"""
Linear regression that finds game's result (user score) based on the game accuracy. The score is {0(loss), 0.5(draw), 1(win)} but we 
make it continuous between 0 and 1.
"""


def linear_regression(scores, accuracies):
    data = {'score': scores,
            'accuracy': accuracies}
    df = pd.DataFrame(data)
    scaler = MinMaxScaler()
    df['score'] = scaler.fit_transform(df[['score']])
    plt.scatter(df['accuracy'], df['score'])
    plt.xlabel('Accuracy')
    plt.ylabel('Score')
    plt.title('Scatter Plot of Accuracy vs. Score')
    plt.show()

    # Instantiate the model
    model = LinearRegression()

    # Fit the model to the data
    X = df[['accuracy']]
    y = df['score']
    model.fit(X, y)

    # Get the coefficients
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    plt.scatter(df['accuracy'], df['score'])
    plt.plot(df['accuracy'], model.predict(X), color='red', linewidth=2)
    plt.xlabel('Accuracy')
    plt.ylabel('Score')
    plt.title('Linear Regression: Accuracy vs. Score')
    plt.show()
    # Example prediction
    new_accuracy = 75
    predicted_score = model.predict([[new_accuracy]])[0]
    print(f"Predicted Score for Accuracy {new_accuracy}: {predicted_score}")

    # Make predictions
    y_pred = model.predict(X)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse}")

# K-Nearest Neighbors (K-NN)


def knn(accuracies):
    accuracies = np.array(accuracies).reshape(-1, 1)

    # Instantiate k-NN model
    k = 3
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(accuracies)

    # Example: Find neighbors for a new data point
    new_data_point = np.array([[72]])
    distances, indices = knn.kneighbors(new_data_point)

    # Visualize the data points and neighbors
    plt.scatter(range(len(accuracies)), accuracies, label='Data Points')
    plt.scatter(indices, accuracies[indices],
                color='red', marker='X', label='Nearest Neighbors')
    plt.scatter(len(accuracies), new_data_point,
                color='green', marker='o', label='New Data Point')

    # Highlight the new data point
    plt.xlabel('Data Points')
    plt.ylabel('Accuracy')
    plt.title('k-NN Visualization of Accuracy Data')
    plt.legend()
    plt.show()

# K-means clustering


def k_means(accuracies):

    accuracies = np.array(accuracies).reshape(-1, 1)
    # Choose the number of clusters (K)
    num_clusters = 6

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters)

    clusters = kmeans.fit_predict(accuracies)

    # Visualize the clusters
    plt.scatter(range(len(accuracies)), accuracies,
                c=clusters, cmap='viridis')
    plt.xlabel('Data Points')
    plt.ylabel('Accuracy')
    plt.title('K-Means Clustering of Accuracy Data')
    plt.show()

# kernel density estimation


def kde_estimation(accuracy_data):
    # Reshape data for scikit-learn input
    accuracy_data = np.array(accuracy_data).reshape(-1, 1)

    # Fit the kernel density estimator
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(accuracy_data)

    # Generate points for the x-axis

    x = np.linspace(min(accuracy_data), max(
        accuracy_data), 1000).reshape(-1, 1)

    # Calculate the log probability density function
    log_dens = kde.score_samples(x)

    # Plot the data and the KDE
    plt.hist(accuracy_data, bins=10, density=True,
             alpha=0.5, color='blue', edgecolor='black')
    plt.plot(x, np.exp(log_dens), color='red', linewidth=2)
    plt.title('Kernel Density Estimation for Chess Accuracy Data')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.show()

    # Create a KDE plot
    sns.kdeplot(accuracy_data, fill=True)

    # Add labels and title
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.title('Kernel Density Estimation of Accuracy')
    plt.show()

# kernel density estimation

# kolmogorov-smirnov test


def kolmogorov_smirnov(accuracies1, accuracies2):
    # Perform the KS test
    ks_statistic, ks_p_value = ks_2samp(accuracies1, accuracies2)
    # Interpret the results
    print(f"KS Statistic: {ks_statistic}")
    print(f"P-value: {ks_p_value}")

    # Determine significance level (alpha) for interpretation
    alpha = 0.05
    if ks_p_value < alpha:
        print("The samples are significantly different.")
    else:
        print("There is no significant difference between the samples.")

# General descriptive statistics like mean etc.


def descriptive_stats(games):
    # Assuming 'games' is your list of game dictionaries

    # 1. Calculate Average Accuracy
    average_accuracy = sum(
        game['accuracy'] for game in games if game["accuracy"] is not None) / len(games)

    # 2. Calculate Average Rating
    average_rating = sum(game['rating'] for game in games) / len(games)

    # 3. Count Wins and Losses
    wins = sum(1 for game in games if game['score'] == 1)
    losses = sum(
        1 for game in games if game['score'] == 0)

    # 4. Score Distribution
    score_distribution = {}
    for game in games:
        score_distribution[game['score']] = score_distribution.get(
            game['score'], 0) + 1

    # 5. Ranking Analysis
    rank_distribution = {}
    for game in games:
        rank_distribution[game['rank']] = rank_distribution.get(
            game['rank'], 0) + 1

    print("Average Accuracy:", average_accuracy)
    print("Average Rating:", average_rating)
    print("Wins:", wins)
    print("Losses:", losses)
    print("Score Distribution:", score_distribution)
    print("Rank Distribution:", rank_distribution)


def spearman_correlation(var1, var2):
    # Calculate Spearman rank correlation coefficient
    correlation_coefficient, p_value = spearmanr(var1, var2)

    # Interpret the results
    print(f"Spearman Rank Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")

    # Determine significance level (alpha) for interpretation
    alpha = 0.05
    if p_value < alpha:
        print("The correlation is statistically significant.")
    else:
        print("There is no significant correlation.")


def correlation_matrix(games):
    df = pd.DataFrame(games)
    # Calculate the correlation matrix
    correlation_matrix = df[['accuracy', 'rating']].corr()
    print(correlation_matrix)


def accuracyDistribution(accuracies_normal, accuracies_important, player):
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies_normal, bins=20, edgecolor='black',
             alpha=0.7, label='Usual', color='blue', density=True)
    plt.hist(accuracies_important, bins=20, edgecolor='black',
             alpha=0.7, label='Important', color='orange', density=True)

    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title(f'Accuracy Distribution for Player: {player}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # Shapiro-Wilk test
    shapiro_wilk(accuracies_normal)

# histogram of mean accuracies across rounds


def accuracyHistogram(accuracies):
    num_rounds = 10

    # Create a histogram
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_rounds + 1),
            accuracies, color='blue', alpha=0.7)
    plt.xlabel('Round')
    plt.ylabel('Mean Accuracy')
    plt.title('Histogram of Mean Accuracy Across Rounds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
