from bs4 import BeautifulSoup
import requests
import urllib.request
from urllib.request import urlopen
import json
import re
from datetime import datetime
import pymongo
from mongo import *

# Function to find an object with a specific property value


def find_object_by_property(array, property_name, target_value):
    for obj in array:
        if property_name in obj and obj[property_name] == target_value:
            return obj
    return None

# get ranking


def get_ranking(original_dict, possible_next=None):

    # Sort the dictionary items based on the points in descending order
    sorted_users = sorted(original_dict.items(),
                          key=lambda x: x[1], reverse=True)

    # if we want to know the rank if the user wins next round
    if possible_next:
        for i, (user, points) in enumerate(sorted_users):
            points += 1.0
    # Assign ranks to users with tied ranking
    ranked_dict = {}
    current_rank = 1
    for i, (user, points) in enumerate(sorted_users):
        if i > 0 and points < sorted_users[i - 1][1]:
            current_rank = current_rank + 1
            ranked_dict[user] = current_rank
        else:
            ranked_dict[user] = current_rank

    return ranked_dict

# get this object when the user did not play the round


def get_blank_game_data(user):
    game_data = {"white": False, "score": 0,
                 "accuracy": None, "rating": 0}
    return game_data


def convert_month_to_numeric(month_str):
    month_numeric = datetime.strptime(month_str, "%B").month
    formatted_month = datetime(2000, month_numeric, 1).strftime("%m")
    return formatted_month

# get info on who is black, white and game result


def get_game_data(game, username):
    # Define the regular expression for the entire block
    block_pattern = re.compile(
        r'\[White "(.*?)"\]\n\[Black "(.*?)"\]\n\[Result "(.*?)"\]')
    # Find matches in the larger data
    block_match = block_pattern.search(game["pgn"])
    # Create dictionaries to store results for each player
    game_data = {}
    # Extract player names and result if a match is found
    if block_match:
        white_player = block_match.group(1).lower()
        black_player = block_match.group(2).lower()
        result = block_match.group(3)
        # Extract scores from the result string
        scores = result.split('-')
        if scores[0] == "1/2" or scores[1] == "1/2":
            white_score = 0.5
            black_score = 0.5
        else:
            white_score = float(scores[0])
            black_score = float(scores[1])
        # Store scores in dictionaries
        score_to_record = white_score if white_player == username else black_score
        accuracy_to_record = None
        if "accuracies" in game:
            accuracy_to_record = game["accuracies"]["white"] if white_player == username else game["accuracies"]["black"]
        else:
            accuracy_to_record = None
        isWhite = True if username == white_player else False
        rating = game["white"]["rating"] if isWhite else game["black"]["rating"]

        game_data = {"white": isWhite, "score": score_to_record,
                     "accuracy": accuracy_to_record, "rating": rating}

    else:
        print("Pattern not found in the larger data.")
    return game_data


# sample web page
tournaments_web_page = 'https://www.chess.com/tournament/live/titled-tuesdays?&page='
api_endpoint = "https://api.chess.com/pub/tournament/"
game_archives_endpoint = "https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}"
n_pages = 5

header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
          'AppleWebKit/537.11 (KHTML, like Gecko) '
          'Chrome/23.0.1271.64 Safari/537.11',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
          'Accept-Encoding': 'none',
          'Accept-Language': 'en-US,en;q=0.8',
          'Connection': 'keep-alive'
          'allow_redirects=False'
          }

tournaments_class = "tournaments-live-name"

tournaments = []
for p in range(7):
    # call get method to request that page
    page = requests.get(tournaments_web_page + str(p))
    # with the help of beautifulSoup and html parser create soup
    soup = BeautifulSoup(page.content, "html.parser")
    tournament_page = soup.find_all('a', {"class": tournaments_class})
    tournaments.append(tournament_page)

for tt, t_page in enumerate(tournaments):
    for t_index, t in enumerate(t_page):
        # total games and total available accuracy information for each game
        href = t["href"]
        # take the first page of players (25 top players in the tournament)
        results_href = href + "?&players=1"
        index = href.rfind("/")
        tournament_name = href[index+1:]
        # tournament api_endpoint to access
        url = api_endpoint + href[index+1:]

        # we need to parse month and year from the tournament
        match = re.search(r'-(\w+)-(\d+)-(\d+)-', href)
        if match:
            month_str = match.group(1)
            day = int(match.group(2))
            year = int(match.group(3))
            month_numeric = convert_month_to_numeric(month_str)
            print("Month (numeric):", month_numeric)
            print("Day:", day)
            print("Year:", year)

        res = requests.get(
            url, headers=header, allow_redirects=True)
        tournament_general = res.json()
        for player in tournament_general["players"]:
            player_doc = {"username": player["username"], "games": []}
            insertIfNotExist(players, "username",
                             player["username"], player_doc)
        players_dict = {player_d['username']: []
                        for player_d in tournament_general["players"]}
        games_dict = {player_d['username']: ''
                      for player_d in tournament_general["players"]}

        # get all tournament games (1-25 place)
        page = requests.get(results_href)
        # with the help of beautifulSoup and html parser create soup
        results_page = BeautifulSoup(page.content, "html.parser")

        user_rows = results_page.find_all(
            'tr', {"class": "tournaments-live-view-results-row"})
        # delete header row
        user_rows = user_rows[1:]
        # for each round
        for r in range(11):
            # for each user
            for u_index, user_row in enumerate(user_rows):
                user = user_rows[u_index].find_all(
                    'a', {"class": "user-username-component"})
                user_ = tournament_general["players"][u_index]["username"]
                archives_endpoint_player = game_archives_endpoint.format(
                    username=user_, YYYY=year, MM=month_numeric)
                round_result_class = "tournaments-live-view-player-result"
                rounds = user_row.find_all(
                    "a", {"class": "tournaments-live-view-player-result"})
                # select the round
                if r >= len(rounds):
                    round_game_link = None
                else:
                    round = rounds[r]
                    round_game_link = round["href"]
                # get the game from archive endpoint
                monthly_games = requests.get(
                    archives_endpoint_player, headers=header)
                monthly_json = monthly_games.json()
                if round_game_link is None or "games" not in monthly_json:
                    game_data = get_blank_game_data(user_)
                else:
                    game = find_object_by_property(
                        monthly_json["games"], "url", round_game_link)
                    game_data = get_game_data(game, user_)

                game_data["tournament"] = tournament_name
                game_data["round"] = r + 1
                # if it's the first round, just add points to 0, otherwise add points to previous points
                if r == 0:
                    game_data["points"] = game_data["score"]
                else:
                    # collection, username, array_field, tournament_index, game_index
                    previous_game = get_element_at_index(
                        players, user_, "games", r - 1)
                    game_data["points"] = previous_game["points"] + \
                        game_data["score"]
                players_dict[user_] = game_data["points"]
                games_dict[user_] = game_data
                # calculate ranking when we reached the last player
                if u_index == len(user_rows) - 1:
                    ranked_list = get_ranking(players_dict)
                    # if penultimate rank, ask for hypothetical ranking
                    ranked_list_possible = get_ranking(
                        players_dict, possible_next=True)
                    for player in tournament_general["players"]:
                        games_dict[player["username"]
                                   ]["rank"] = ranked_list[player["username"]]
                        games_dict[player["username"]
                                   ]["possible_rank"] = ranked_list_possible[player["username"]]

                        pushArray(players, player["username"], "games",
                                  games_dict[player["username"]])
