from g2048 import Game2048
import neat
import time, sys
import pickle
import argparse
import os
import random
import numpy


def ai_suggest_move(net, game, headless = False):
    '''
    a method which has the ai suggest a move in 2048 and trys the move. If it is not 
    valid, valid_moves is updated in the game. this method then uses valid_moves to filter out invalid moves from the Ai's suggestion. 
    I have some concerns about how this will affect the learning of the game, but can't come up with a better way to teach this nn not to TRY THINGS THAT ARE ILLEGAL. dumb boy. he's a friend, though.
    
    Attributes:
    
    Returns
    move(int): a 2, 4, 6, or an 8 that maps to move down, move left, move right, and move up respectively. 
    -1 (int): a marker that will tell the game that the AI can't make any more moves and to trigger end-game. 
    
    '''
    output = net.activate(game.board.ravel())
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

def empties_state(last_board, new_board):
    '''
    gives a value based on the improvement of board states, such that 
    maximizing open space will give heightened fitness. 
    Attributes:
    last_board (np.array): board from previous game state
    new_board (np.array): board after move was made

    Returns:
    float or 0: if new board is better (more open space) than old board, it gives a small bonus to fitness. 
    
    '''
    new_empties = len(np.where(new_board == 0))- len(np.where(last_board == 0))
    if new_empties > 0:
        return new_empties/10000 # May be to large or too small. Major concern is that it might go over one. actually...if the game goes on for 400 turns, it will. So that's...worth thinking about. 
    else:
        return 0

def gameplay_eval(genomes, config, headless = True):
    '''
    a method which will determine how good the AI is doing on the game. 
     
    '''
    
    for genome_id, genome in genomes:
        fit = 0
        net = neat.nn.RecurrentNetwork.create(genome, config)

        #have an AI try the game, record its score metrics. 
        game = Game2048(ai = True, headless = True)
        while game.game_over == False:
            last_board = game.board
            last_score = game.score
            
            empties1 = np.where(game.board == 0)[0] #np.where returns a tuple with an array inside it, for some reason!?!
            get_from_ai = ai_suggest_move(net, game, headless = True) #must be int 2, 4, 6 or 8, related to moves down, left, right and up respectively
            if get_from_ai == -1:
                break
            
            game.get_move(ai_move = get_from_ai)
            valid = game.game_step()
            if valid == -1:
                continue
            new_board = game.board
            updated_score = game.score
            fit += empties_state(last_board, new_board)
            
            
            
        genome.fitness = (.3*fit)+ (.7*game.score/44000) #fitness is a combination of the sum of empty tiles made every turn and the score. 
        #print(genome.fitness)                               
        
        # print(genome.fitness, genome_id)      #handles that show best guy.        



def train_ai(config_path):
    """
    runs the NEAT algorithm to train a neural network to play 2048
    :param config_file: location of config file
    :return: winnner (NEAT genome file) best genome that the algorithm has developed 
    """
    # winner. uhm...seems to be the wrong guy. Maybe only the best from the latest algo?
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    #print('now running for winner step')
    winner = p.run(gameplay_eval, 50)
    print('\nBest genome:\n{!s}'.format(winner))

    # show final stats
    #print(pop.statistics.best_genome())
    return winner


parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument("-h", required = False, default =True, help='trian the net in Headless mode')
parser.add_argument('-c', required=True, help='path of the config file')

args = parser.parse_args()
h = args.h

if __name__ == "__main__":
    winner = train_ai(config_path)