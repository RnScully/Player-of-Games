import time
import neat
from IPython.display import clear_output
from g2048 import Game2048
from train_2048 import ai_suggest_move
import pickle
import argparse

def display_skills(saved_ai, config_path):
    '''
    A function that will have one AI play the game and display moves
    '''
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    ai = neat.nn.RecurrentNetwork.create(saved_ai, config)

    game = Game2048(ai = True, headless = False)
    game.show_board()
    while game.game_over == False:
            
            last_score = game.score
            
            
            get_from_ai = ai_suggest_move(ai, game, headless = False) #must be int 2, 4, 6 or 8, related to moves down, right, left and up respectively
            game.get_move(ai_move = get_from_ai)
            game.game_step()
            
            #time.sleep(.4) #if you want it in realtime. For checking to see if is working, you want it all at once
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A tutorial of argparse!')
    parser.add_argument("-m", required = True, help='model to display')
    parser.add_argument('-c', required = True, help = 'neat-python config file is needed to rebuild the ai from the save')

    args = parser.parse_args()
    config_path = args.c
    saved_ai = args.m

    display_skills(saved_ai, config_path)
