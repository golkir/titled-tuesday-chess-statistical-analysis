import re
from datetime import datetime
from mongo import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


import tools


def avgPartial(arr, include=None, exclude=None, every_nth=None, onlyIfChance=False, minRank=None):
    if exclude is not None and include is not None:
        raise ValueError(
            "Only one of included or excluded elements should be specified")

    if exclude:
        data = [element for i, element in enumerate(
            arr) if (i + 1) % every_nth != 0 and element["accuracy"] is not None]
    if include and not onlyIfChance:
        data = [element for i, element in enumerate(
            arr) if (i + 1) % every_nth == 0 and element["accuracy"] is not None]
    if include and onlyIfChance:
        data = [element for i, element in enumerate(
            arr) if (i + 1) % every_nth == 0 and arr[i-1]["possible_rank"] < minRank and element["accuracy"] is not None]

    data = [el["accuracy"] for el in data]

    return (sum(data) / len(data), len(data))


def analyze(username="magnuscarlsen"):
    # get hikaru
    player = getPlayer(players, username)
    games = player["games"]

    # calculate mean accuracy in all rounds except final

    rounds = 10
    avgs_not_last = []

    for i in range(10):
        # calculate mean accuracy for each round in all tournaments except 11
        round_avg, *_ = avgPartial(games, include=True, every_nth=i+1)
        avgs_not_last.append(round_avg)

    avg_last, *_ = avgPartial(games, include=True, every_nth=11)
    avgs_not_last = np.array(avgs_not_last)
    avgs_if_win, *_ = avgPartial(
        games, include=True, every_nth=11, onlyIfChance=True, minRank=5)

    accuracies = [game['accuracy']
                  for game in games if game['accuracy'] is not None]
    accuracies_final = [{"accuracy": game['accuracy'], "index": i} for i, game in enumerate(
        games) if (i + 1) % 11 == 0 and game['accuracy'] is not None and games[i-1]["possible_rank"] < 5]

    accuracies_final = [item["accuracy"]
                        for i, item in enumerate(accuracies_final)]

    scores = [game['accuracy'] for i, game in enumerate(
        games) if game['score'] is not None and game['accuracy'] is not None]

    print("avgs general")
    print(len(accuracies))

    print("avgs important")
    print(len(accuracies_final))

    # Descriptive statistics

    tools.descriptive_stats(games)

    # Draw histogram of accuracies in 1-10 rounds
    tools.accuracyHistogram(avgs_not_last)

    # visually evaluate accuracies distribution in all normal games and all important games (if player has winning chances in the last round)

    tools.accuracyDistribution(accuracies, accuracies_final, username)

    # Kernel density estimation
    tools.kde_estimation(accuracies)

    # Kolmogorov-Smirnov test

    tools.kolmogorov_smirnov(accuracies, accuracies_final)

    # Spearman correlation between game scores and accuracy in the game

    tools.spearman_correlation(scores, accuracies)

    # calculate correlation matrix
    tools.correlation_matrix(games)

    # Maximum Likelihood estimation to evaluate if game parameters (score, accuracy, etc.) follow normal distribution

    tools.maximum_likelihood_estimation(accuracies)

    # K-Means clustering
    tools.k_means(accuracies)

    # K-nearest neighbors
    tools.knn(accuracies)

    # Linear regression
    tools.linear_regression(scores, accuracies)


analyze()
