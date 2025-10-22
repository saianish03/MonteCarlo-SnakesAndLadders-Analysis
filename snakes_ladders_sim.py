import numpy as np
import random
from collections import defaultdict
from pprint import pprint
from functools import cache
import matplotlib.pyplot as plt
from tqdm import tqdm

class SnakesAndLadders:
    
    def __init__(self):
        self.start_position = 0
        self.snakes = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 94: 73, 95: 75, 98: 78}
        self.ladders = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
        self.rolls_frequency = defaultdict(int) # {num_rolls_to_win: frequency}
        self.snake_hits = defaultdict(int) # {snake_head_square: hit_count}
        self.ladder_hits = defaultdict(int) # {ladder_bottom_square: hit_count}
        self.rolls_to_win = [] # roll counts to win for each game
        self.total_games = 0 # counter to count number of rolls
        self.most_likely_rolls = -1 # to track most likely rolls
        self.convergence_threshold = 0.003
        self.converging_game_number = float('inf')
        self.min_game_example = None
        self.high_game_example = None
        self.typical_game_example = None
        self.high_threshold = 0.95 

    def roll_die(self):
        """Roll a standard 6-sided die and return the result"""
        return random.randint(1, 6)
    
    def next_position(self, position, dice_roll):
        """Calculate the next position after rolling the die, applying snake/ladder effects"""
        position += dice_roll
        if position in self.ladders:
            self.ladder_hits[position] += 1
            position = self.ladders[position]

        elif position in self.snakes:
            self.snake_hits[position] += 1
            position = self.snakes[position]
            
        return position
    
    def play_game(self):
        """Play a complete game from start to finish and return the number of rolls needed"""
        position = self.start_position
        rolls = 0
        path = [position]
        
        while position < 100:
            dice_roll = self.roll_die()
            rolls += 1
            position = self.next_position(position, dice_roll)
            path.append(position)
        
        return rolls, path

    @cache
    def dp_for_min_rolls(self, position):
        """Calculate the minimum number of rolls needed to reach position 100 from a given position using dynamic programming"""
        if position >= 100:
            return 0
        min_moves = float('inf')
        for i in range(1, 7):
            if position + i in self.snakes:
                continue
            min_moves = min(min_moves, 1 + self.dp_for_min_rolls(self.next_position(position, i)))
        return min_moves

    def is_converging(self, prob_history, window_size=50, min_games=500):
        """
        Check if the simulation has converged using a rolling window approach
        """

        # dont check until min games met
        if len(prob_history) < min_games:
            return False
        
        # dont check until window size met
        if len(prob_history) < window_size:
            return False
        
        # get most recent probs 
        recent_probs = prob_history[-window_size:]
        
        # Calculate the range of recent estimates
        min_recent = min(recent_probs)
        max_recent = max(recent_probs)
        
        if min_recent == 0: # edge case found during simulations
            return False
        
        # check if the range is within the threshold
        relative_range = (max_recent - min_recent) / min_recent
        
        return relative_range < self.convergence_threshold

    def run_simulation(self, n_games=10000, stop_at_convergence: bool = False):
        """Run multiple games of snakes and ladders to collect statistical data"""
        
        print(f"\nRunning {n_games} games of snakes and ladders...\n")

        prob_history = [] 
        
        for game_num in tqdm(range(n_games)):
            rolls_needed, path = self.play_game()

            self.rolls_to_win.append(rolls_needed)
            self.rolls_frequency[rolls_needed] += 1
            
            # calculating most likely rolls:
            if self.most_likely_rolls == -1 \
                or self.rolls_frequency[self.most_likely_rolls] < self.rolls_frequency[rolls_needed]:
                self.most_likely_rolls = rolls_needed
            
            self.total_games += 1

            current_rolls = rolls_needed
            current_path = path

            # games matching min rolls
            if (self.min_game_example is None or current_rolls < self.min_game_example[0]):
                self.min_game_example = (current_rolls, current_path.copy())
            
            # games matching most likely rolls
            if (current_rolls == self.most_likely_rolls and 
                self.typical_game_example is None):
                self.typical_game_example = (current_rolls, current_path.copy())
            
            # capture high roll game example
            if (self.high_game_example is None or current_rolls > self.high_game_example[0]):
                self.high_game_example = (current_rolls, current_path.copy())

            prob_most_likely = self.rolls_frequency[self.most_likely_rolls] / self.total_games
            prob_history.append(prob_most_likely)  # add to history for convergence checking
            
            # check convergence AFTER calculating the prob
            if self.is_converging(prob_history):
                # set convergence point once (first time we find it)
                if self.converging_game_number == float('inf'):
                    self.converging_game_number = self.total_games
                
                if stop_at_convergence:
                    print("Stopping at Convergence at Game Number: ", self.total_games)
                    break
        
        print("\nSimulation completed!!")

    def compute_stats(self):
        """Calculate and display statistical analysis of the simulation results"""
        # q1:
        avg = sum(self.rolls_to_win) / self.total_games

        # q2:
        sim_min_rolls = min(self.rolls_to_win)
        sim_max_rolls = max(self.rolls_to_win)

        # q3:
        sorted_rolls = sorted(self.rolls_to_win)
        median = (sorted_rolls[(self.total_games-1)//2] + sorted_rolls[self.total_games//2]) / 2
        most_likely_rolls = self.most_likely_rolls
        # plot shape of distribution !!!

        # q4:
        prob_most_likely = self.most_likely_rolls / self.total_games
        # q6:
        print("\nQ1)")
        print("Average rolls: ", int(avg))
        print("\nQ2)")
        print("Theorietical max rolls: ", float('inf'))
        print("Theorietical min rolls:", self.dp_for_min_rolls(0))
        print("Sim max rolls: ", sim_max_rolls)
        print("Sim min rolls: ", sim_min_rolls)
        print("\nQ3)")
        print("Average rolls: ", avg)
        print("Median rolls: ", median)
        print("Most likely rolls: ", most_likely_rolls)
        print("\nQ4)")
        print("Most likely rolls: ", most_likely_rolls)
        print("Probability of most likely rolls: ", prob_most_likely)
        print("\nQ5)")
        print("Converging Game Number: ", self.converging_game_number)
        print("\nQ6)")
        print("Are all snakes and all ladders equally stepped on??") # answer this
        print("Which snake is most frequently stepped on: ", max(self.snake_hits, key=self.snake_hits.get))
        print("Which ladder is most frequently stepped on: ", max(self.ladder_hits, key=self.ladder_hits.get))
        print()


if __name__=="__main__":
    sim = SnakesAndLadders()
    sim.run_simulation(n_games=10000, stop_at_convergence=False)
    sim.compute_stats()
    plt.bar(sim.rolls_frequency.keys(), sim.rolls_frequency.values(), color = 'blue')
    plt.xlabel('roll number')
    plt.ylabel('freq')
    plt.show()
