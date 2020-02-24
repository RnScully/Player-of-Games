from g2048 import Game2048
import neat
import time
import pickle




def gameplay_eval(genomes, config):
'''
a method which will determine how good the AI is doing on the game. 
I propose from a first pass a minimization of tiles on the board. '''

    population_size = 50 # set number of games to run to find best player
    induviduals = []# fill with neat generated n population_size 
    
	for _ in range(population_size):
        #have an AI try the game, record its score metrics. 
        game = Game2048(ai = True)
        while game.game_over = False
            give_to_ai = game.board.ravel()
            last_score = game.score()
            empties = np.where(game.board == 0)[0]
            get_from_ai = ai_suggest_move(give_to_ai)
            game.get_move(get_from_ai)



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
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(gameplay_eval, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))