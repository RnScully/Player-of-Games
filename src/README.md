# Welcome to the src!

Here you will find some scripts!

### ai_demo.py
this script will run either type of ai, NEAT or Q-Learning, by loading their models and asking for moves each game. It does not train them or penalize them for mistakes, simply runs through a game. 

it contains some helperfunctions for getting statistics out of each model type. 

You can run it by calling python ai_demo.py with the arg -m Q or -m N (Q will run a 2000-game trained Q-Learner, N will give one of my best NEAT decision makers)

Args are:
- -m : the sort of model to debut. N for neat, Q for Q-learning
- -c : the path to the custom neat configuration that python-neat needs to build its Nueral net
- -nm : the path to the python-neat neural net save file. 
- -qm : the path to the Q-learning model to run. 
- -s : the time, in fractions of a second, to rest between moves. 

### g2048.py

This is the game object, containing the Game2048 class and some script to run the game, the rules for the game as game-mechanics, and some special rules made for training the ai. 

### neat_2048.py

This script evolves NEAT neural nets to play the game. 

Args are:
- -v : Bool, true will run the game in headless mode
- -c : the path to the custom neat configuration that python-neat needs to build its Nueral net
- -p : include -p to parallelize the training of the neats. It vastly speeds of the process
- -n : the name you want to save the final model as
- -g : how many generations you want the evolvers to get. 

### q_learn_2048.py
This script trains a Q-Learner using tensorflow Nueral Nets. 

Args are:
- -v : Bool, true will run in headless mode
- -m : string, path of model to load, if not passed it starts from an untrained random model 
- -d : int, training depth, how far back in the game history to train a good or bad move (note that invalid moves are set to always train on just the last 2 moves)
- -n : string, name to save the new model as
- -g : int, number of games to run the model over

### tools.py
An assortment of helperfunctions I used in this project. Thanks to Code_Reclaimers for draw_net(), it was very helpful in this project. 

