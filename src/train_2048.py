from g2048 import Game2048
import neat
import time
import pickle

def ai_suggest_move(net, board_state):
    output = net.activate(board_state)
    values = [2, 4, 6, 8]
    d = dict(zip(output, values))
    move = d.get(max(d.keys()))
#     if move == 2: #this section outputs the move for visuals and debugging
#         print('slide down')
#     elif move == 4:
#         print('slide left')
#     elif move == 6:
#         print('slide right')
#     else:
#         print('slide up')
    return move

def gameplay_eval(genomes, config):
    '''
    a method which will determine how good the AI is doing on the game. 
    I propose from a first pass a minimization of tiles on the board. '''
    population_size = 50 # set number of games to run to find best player
    induviduals = []# fill with neat generated n population_size 
    
    for genome_id, genome in genomes:
        net = neat.nn.RecurrentNetwork.create(genome, config)

        #have an AI try the game, record its score metrics. 
        game = Game2048(ai = True, headless = True)
        while game.game_over == False:
            give_to_ai = game.board.ravel()
            last_score = game.score
            empties = np.where(game.board == 0)[0] #np.where returns a tuple with an array inside it, for some reason!?!
            get_from_ai = ai_suggest_move(net, give_to_ai) #must be int 2, 4, 6 or 8, related to moves down, right, left and up respectively
            game.get_move(ai_move = get_from_ai)
            game.game_step()
            updated_score = game.score
            genome.fitness = game.score/44000 #fitness for now is just the easiest thing to measure because I already built the score method. 
                                                #THIS IS PROBABLY NOT A GOOD FITNESS METRIC
        
        # print(genome.fitness, genome_id)      #handles that show best guy.        



def train_ai(config_file):
    """
    runs the NEAT algorithm to train a neural network to play 2048
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    #p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    #print('now running for winner step')
    winner = p.run(gameplay_eval, 50)
    #print(winner)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))