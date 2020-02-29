import time
import neat
from IPython.display import clear_output
from g2048 import Game2048
from train_2048 import ai_suggest_move
import pickle
import argparse

def display_skills(saved_ai, config_path, tokenized = False):
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
            
            
            get_from_ai = ai_suggest_move(ai, game, headless = False, tokenize = tokenized) #must be int 2, 4, 6 or 8, related to moves down, right, left and up respectively
            game.get_move(ai_move = get_from_ai)
            game.game_step()
            
            #time.sleep(.4) #if you want it in realtime. For checking to see if is working, you want it all at once
def handle_random_move(game, headless = False):
    '''
    a method which has the ai suggest a move in 2048 and trys the move. If it is not 
    valid, valid_moves is updated in the game. this method then uses valid_moves to filter out invalid moves from the Ai's suggestion. 
    I have some concerns about how this will affect the learning of the game, but can't come up with a better way to teach this nn not to TRY THINGS THAT ARE ILLEGAL. dumb boy. he's a friend, though.
    
    Attributes:
    
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

def random_play(num_runs):
    
    
    scores = []
    best_tiles = []
    empties = []
    for _ in range(num_runs):
        empty = 0
        game = Game2048(ai = True, headless = True)

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
    return (scores, best_tiles, empties)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A tutorial of argparse!')
    parser.add_argument("-m", required = True, help='model to display')
    parser.add_argument('-c', required = True, help = 'neat-python config file is needed to rebuild the ai from the save')

    args = parser.parse_args()
    config_path = args.c
    saved_ai = args.m

    display_skills(saved_ai, config_path)
