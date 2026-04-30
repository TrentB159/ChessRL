import chess
import chess.pgn
import chess.engine
import math
import pandas as pd
import numpy as np
import statistics
from scipy.stats import hmean

# stockfish engine path
# this is the path on my Home machine it should be chnaged if you are using it 
STOCKFISH_PATH = r"C:\Users\p45mi\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

# the input game file and the output CSV file
# MUST BE CHNAGED MANUALLY UPON USE
pgn_path = "HighEloBaller2000/NewElo_test.pgn"
output_csv = "HighEloBaller2000/NewElo_test.csv"

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
results_list = []

# calculate accuracy based of the Lichess method
def volatility_adjusted_accuracy(move_accuracies, win_pcts):
    if len(move_accuracies) < 4:
        return statistics.mean(move_accuracies)

    n = len(move_accuracies)
    
    if n <= 20:
        window_size = 4
    elif n <= 40:
        window_size = 6
    elif n <= 60:
        window_size = 8
    else:
        window_size = 10

    volatilities = []
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, start + window_size)
        start = max(0, end - window_size)
        window_win_pct = win_pcts[start:end]
        std = np.std(window_win_pct)
        volatilities.append(std)

    volatility_weighted_mean = np.average(move_accuracies, weights=volatilities)
    harmonic_mean = hmean(move_accuracies)
    final_accuracy = (volatility_weighted_mean + harmonic_mean) / 2
    return final_accuracy

# calculate accuracy based of the Lichess method
def centipawn_to_win_pct(cp:float):
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)

# calculate accuracy based of the Lichess method
def win_loss_to_accuracy(delta_win_pct):
    return 103.1668 * math.exp(-0.04354 * delta_win_pct) - 3.1669


# loop through each game in the PGN file, before and after each move measure the centipawn score and record
# count Blunders, Mistakes, Inaccuracies, Average Centipawn Loss and Accuracy Score
# write the results to a CSV file at the end
with open(pgn_path) as pgn_file:
    game_number = 0
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        game_number += 1
        board = game.board()

        cp_losses = []
        inaccuracies = 0
        mistakes = 0
        blunders = 0
        accuracies = []
        win_pcts = []

        for move in game.mainline_moves():
            if board.turn != chess.WHITE:
                board.push(move)
                continue

            info_before = engine.analyse(board, chess.engine.Limit(depth=10))
            score_before_cp = info_before["score"].white().score()
            if score_before_cp is None:
                board.push(move)
                continue
            win_pct_before = centipawn_to_win_pct(score_before_cp)

            board.push(move)

            info_after = engine.analyse(board, chess.engine.Limit(depth=10))
            score_after_cp = info_after["score"].white().score()
            if score_after_cp is None:
                continue

            win_pct_after = centipawn_to_win_pct(score_after_cp)
            win_pcts.append(win_pct_after)
            delta_win_pct = (win_pct_before - win_pct_after)

            accuracy = win_loss_to_accuracy(delta_win_pct)
            accuracies.append(accuracy)

            cp_loss = (score_before_cp - score_after_cp)
            cp_losses.append(cp_loss)

            if cp_loss >= 300:
                blunders += 1
            elif cp_loss >= 100:
                mistakes += 1
            elif cp_loss >= 50:
                inaccuracies += 1

        avg_cp_loss = sum(cp_losses) / len(cp_losses) 
        avg_accuracy = volatility_adjusted_accuracy(accuracies, win_pcts)

        results_list.append({
            "GameNumber": game_number,
            "Inaccuracies": inaccuracies,
            "Mistakes": mistakes,
            "Blunder": blunders,
            "AvgSPLoss": round(avg_cp_loss, 2),
            "accuracy%": round(avg_accuracy, 2)
        })

        print(f"Game {game_number}")

engine.quit()

# Save all games results to CSV
df = pd.DataFrame(results_list)
df.to_csv(output_csv, index=False)
