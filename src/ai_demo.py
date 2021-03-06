import time
import neat
from IPython.display import clear_output
from g2048 import Game2048

from IPython.display import clear_output
import sys
import os

from neat_2048 import ai_suggest_move
from q_learn_2048 import Q_Player
import pickle
import argparse

def print_move(x):
    '''
    takes an int, 2, 4, 6 or 8, and prints the move that represents
    Paramaters
    x (int) 2,4,6, or 8, or else nothing will happen. 
    '''
    if x == 6:
        print('slide right')
    if x == 8:
        print('slide up')
    if x ==4:
        print('slide left')
    if x == 2:
        print('slide down')


def demo_QR_learner(ai_path, sleep_time = 0, headless = True):
    '''
    function that runs the AI without training it for showcasing and metrics of the AI at specific training levels. 
    ++++++++++
    Parameters 
    ai_path (str): path of the ai model
    sleep_time (float): how many seconds to wait between turns
    headless (bool): whether or not to run the game in headless mode
    ++++++++++
    Returns
    game.board.max() (int): best tile reached in game
    game.score (int): final score
    
    '''
    agent = Q_Player(model = ai_path, demo = True)
    game = Game2048(ai = True, headless = False, strict = False)

    

    while game.game_over == False:
        
        last_board = game.board
        #print('Game Over: {}'.format(game.game_over))

        move = agent.ai_suggest_move(game)
        #if headless == False:
            #game.show_board()
        #agent.calc_reward(game, move)

        #agent.give_reward(game)

        game.get_move(move)
        os.system('clear')
        game.game_step()
        if headless == False:
            print_move(move)
            print('')

        new_board = game.board
        time.sleep(sleep_time)
        # clear_output(wait = True)
         # cheating because it just adds space and you seem to have a new screen, but uh...time constraints
    return game.board.max(), game.score

def display_neat_skills(saved_ai, config_path, tokenized = False, headless = False):
    '''
    A function that will have one AI play the game and display moves
    ++++++++++
    Parameters
    saved_ai (str): path to python-neat genome you're running
    config_path (str): path to python-neat
    tokenized (bool): wether to use a 16 feature game board(False) or a 256 feature tokenized game board. (True)
    headless (bool): whether to display the screen
    ++++++++++
    Returns
    game.board.max() (int): best tile reached in game
    game.score (int): final score
    '''
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    ai = neat.nn.RecurrentNetwork.create(saved_ai, config)

    game = Game2048(ai = True, headless = False)
    game.show_board()
    while game.game_over == False:
            
            last_score = game.score
            
            
            get_from_ai = ai_suggest_move(ai, game, headless = False, tokenize = tokenized) #must be int 2, 4, 6 or 8, related to moves down, right, left and up respectively
            game.get_move(ai_move = get_from_ai)
            os.system('clear') #I thought I was cheating before, but apparently allof these just print huge ammoungs of empty space. 
            game.game_step()

            if headless == False:
                print_move(get_from_ai) #get from ai is the ai's move
                print('')

            time.sleep(sleep_time) #if you want it in realtime. For checking to see if is working, you want it all at once
       
    return game.board.max(), game.score

def handle_random_move(game, headless = False):
    '''
    a method which has the ai suggest a move in 2048 and trys the move. If it is not 
    valid, valid_moves is updated in the game. this method then uses valid_moves to filter out invalid moves from the Ai's suggestion. 
    I have some concerns about how this will affect the learning of the game, but can't come up with a better way to teach this nn not to TRY THINGS THAT ARE ILLEGAL. dumb boy. he's a friend, though.
    ++++++++++
    Parameters:
    game (g2048 object): game board
    headless (bool): True tells the game to run headless, False shows the game moves
    ++++++++++
    Returns
    move(int): a 2, 4, 6, or an 8 that maps to move down, move left, move right, and move up respectively. 
    -1 (int): a marker that will tell the game that the AI can't make any more moves and to trigger end-game. 
    
    '''
    output = [np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()]
    
    
    values = [2, 4, 6, 8]
    
    
    valid_moves = game.valid_moves
   
    d = dict(zip(np.array(output)[valid_moves], np.array(values)[valid_moves]))
    #print(np.array(['up','left', 'right', 'down'])[valid_moves])
    #print(game.board) #testing prints
    
    if len(d.keys()) == 0:
        #print('ai_suggest_move is out of possible moves')
        game.game_over = True #game is over
        return  -1
    move = d.get(max(d.keys()))
    
    
    if headless == False:
        if move == 2: #this section outputs the move for visuals and debugging
            print('slide down')
        elif move == 4:
            print('slide left')
        elif move == 6:
            print('slide right')
        else:
            print('slide up')
    return move

def random_play(num_runs, headless = True):
    '''
    A function wich runs the game using handle_random_move() to decide what the next move will be, simulates random play for graphs
    ++++++++++
    Parameters
    num_runs (int): how many times you want to trial random play.
    headless (bool):  True tells the game to run headless, False shows the game moves
    ++++++++++
    Returns
    score (lst): list of final scores 
    empties (lst): list of number of empties generated during each game
    best_tiles (lst): list of best tiles reached in each game
    '''
    
    scores = []
    best_tiles = []
    empties = []
    for _ in range(num_runs):
        empty = 0
        game = Game2048(ai = True, headless = headless)

        while game.game_over == False:

                get_from_ai = handle_random_move(game, headless = True) #must be int 2, 4, 6 or 8, related to moves down, right, left and up respectively
                last_board = game.board
                game.get_move(ai_move = get_from_ai)
                game.game_step()
                new_board = game.board
                empty+=empties_state(last_board,new_board)
                

        best_tiles.append(game.board.max())
        scores.append(game.score)
        empties.append(empty)
    return scores, best_tiles, empties

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A tutorial of argparse!')
    parser.add_argument("-m", required = True, help='model to display, Q for Q learner N for NEAT')
    parser.add_argument('-c', default = 'models/config/neat_config_2', help = 'neat-python config file is needed to rebuild the ai from the save')
    parser.add_argument('-s', default = .4, help = "how many seconds to wait between moves")
    parser.add_argument('-nm', default = 'models/tuesday_mk2.sav', help ='neat model to use for the neat display')
    parser.add_argument('-qm', default = 'models/Q2000.h5', help = 'path of qm model to use')     
    args = parser.parse_args()
    
    sleep_time = float(args.s)

    which_ai = args.m
    
    if which_ai.lower() == 'N'.lower():
        #get info for NEAT
        config_path = args.c
        neat_model_path = args.nm
        with open(neat_model_path, 'rb') as pickle_file:
            saved_ai = pickle.load(pickle_file)
        
    
    else: #oh ho ho, don't do error trapping? that's lazy coding for you
        ai_path = args.qm

    
    

    if which_ai.lower() == 'N'.lower():
        display_neat_skills(saved_ai, config_path, sleep_time)
    else:
       demo_QR_learner(ai_path, sleep_time = sleep_time, headless = False)