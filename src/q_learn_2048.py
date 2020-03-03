

from g2048 import Game2048
import numpy as np

from g2048 import Game2048
from train_2048 import empties_state
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

class RL_Player():
    def __init__(self, reward_depth = 5, model = None, demo = False, headless = False):
        self.reward = 0
        self.gamma = 0.9
        self.long_memory =[]
        # self.dataframe = pd.DataFrame()
        self.oldest_mem = 0
        self.target = []
        self.X = []
        self.learning_rate = .005
        self.reward_depth = reward_depth
        if model == None:
            self.model = self.neural_net()
        else:
            self.model = keras.models.load_model(model)
        self.headless = headless 
        self.demo = demo
        self.bad_move = 0
        self.memory = [[] for _ in range(self.reward_depth)]
        self.debug = 0
        self.debug1 = 0
        self.debug2 = -1
    
    def tokenize_board(self, board):
        '''method takes the game board and tokenizes all the arrays into 16 other arrays for if there's a 2, a 4, an 8, etc in a location to just a 1. hopefully...makes matching better
        Attributes 
        game( game2084 object): the game. 
        Returns
        tokenized 256x1 tokenization of all* possible game states. 
        
        * does not account for the 136k tile. I just..don't expect to need that. Frankly, 16k and 32k seem a reach
        '''
        tokenized = np.array([])
        
        for x in board.ravel():
            blank = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            if x != 0:
                blank[int(np.log2(x))] = 1
            tokenized = np.append(tokenized, blank)
        return tokenized
    
    def clear_memory(self, game):
        if game.game_over == True:
            self.memory = [[] for _ in range(self.reward_depth)]
            self.long_memory = []
            self.oldest_mem = 0
    
    def ai_suggest_move(self, game):
        '''
        method which takes the game and the model (the ai we are training) and suggests a move.
        
        Attributes:
        Game(g2048 object): the game the AI is currently playing
        
        '''
        self.bad_move = 0
        board = game.board
        
        # tokenize_board block takes the game board and tokenizes all the arrays into 16 other arrays for if there's a 2, a 4, an 8, etc in a location to just a 1. hopefully...makes matching better

        tokenized = self.tokenize_board(board) 
        
        pred= self.model.predict(tokenized.reshape(1,-1))
        output = [i for i in pred[0]]
        
        values = [2, 4, 6, 8]
        valid_moves = game.valid_moves
        
        

        d = dict(zip(np.array(output)[valid_moves], np.array(values)[valid_moves])) # all valid moves
        
        
        
        if self.demo == False: # only have random mvoe interference and gambling in learning rounds. 
            if np.random.randint(50+len(game.history)) < 3 and len(list(d.values())) > 0: # 5% of the time in the early game and decreasing as game length goes on, take random gamble moves to hopefully learn new tactics
                rand_move = random.choice(list(d.values()))
                self.memory.pop(0)
                output.append(rand_move)
                self.memory.append(np.array(output).reshape(1,-1))
                if self.headless == False: 
                    print('random Move')
                return rand_move
        
        if len(d.keys()) == 0:
            #print('ai_suggest_move is out of possible moves')
            game.game_over = True #game is over
            return  -1

        if game.strict == True:
            if max(output) in d.keys(): #if the suggested move is not in valid_moves, it wont be in d, and so this will be negative, and the game will end
                move = d.get(max(d.keys()))
                self.memory.pop(0)
                output.append(move)
                self.long_memory.append(np.array(output).reshape(1,-1))
                self.memory.append(np.array(output).reshape(1,-1))
                return move
            else:
                game.game_over = True #game is over because this setting does not allow invaild moves:
                return  -1
        else:
            if max(output) not in d.keys():
                self.bad_move = -1 #triggers bad move if the thing has to take the next best move. 
                self.memory.pop()
                self.memory.insert(0, self.oldest_mem) #puts the oldest memory back on because invalid moves do not update board
            move = d.get(max(d.keys())) #takes the next best move that is valid
            self.oldest_mem = self.memory.pop(0)
            output.append(move)
            self.long_memory.append(np.array(output).reshape(1,-1))
            self.memory.append(np.array(output).reshape(1,-1))
            
            return move
    
    def calc_reward(self, game, move):
        '''
        Runs right after AI suggest move
        checks if reinforcement is merited after the last move, if so, updates the fit of model. 
        Attributes
        game (Game2048 object)
        move (int): int passed from ai_suggest_move()
        
        Updates reward
        
        '''
        self.reward = 0

        
        ## get old board and new board by checking move against the game's built in moves. 
        old_board = game.board
        
        if move == 2: 
            next_board = game.slide_down()
        elif move == 4:
            next_board = game.slide_left()
        elif move == 6:
            next_board = game.slide_right()
        elif move == 8:
            next_board = game.slide_up()
        else:
            next_board = np.array([[ 1,  1,  1,  1],  #simple full board for end-game board state comparison. 
                                   [ 1,  1, 1, 1],
                                    [ 1, 1, 1,  1],
                                   [ 1,  1,  1,  1]])
        
        tiles_combined = empties_state(old_board, next_board)
        
        
        big_tiles = dict({32:6, 64: 6, 128: 10, 256: 10, 512: 20, 1024: 30, 2048: 100, 4096: 1000})
        if np.amax(old_board) < np.amax(next_board): # checks to see if the bot has combined tiles or gotten a big one. 
            if np.amax(game.board) in big_tiles.keys():
                self.reward += big_tiles[np.amax(game.board)]*2
                if self.headless == False:    
                    print('big tiles')
            else: 
                self.reward +=4 
                if self.headless == False:
                    print('biggest tile yet')
        else:
            a = list(old_board.ravel())
            b = list(next_board.ravel())
            for i in big_tiles.keys():
            
                if b.count(i) - a.count(i) > 0:
                    self.reward+= big_tiles[i]
                    if self.headless == False:
                        print('big tiles')
        
        
        
        if tiles_combined > 2:
            self.reward += 1
            if self.headless == False:
                print('board managment bonus!')
        
        elif self.bad_move == -1 and len(game.history) > 10: # doesn't start penalizing for bad moves untill after a few turns of play
            self.reward += -2
            if self.headless == False:
                print('invalid move')
            
        if game.game_over == True:
            self.reward = -30
            if self.headless == False:
                print('game over penalty')
        self.bad_move == 0 # Reset bad move!
        old_score = game.score
        
    def give_reward(self, game):
        '''if there is a reward, this will get the game history and the memory of the outputs, multiply the outputs by the reward and its time discount
        
        '''
        values = [2, 4, 6, 8]
        discount = 1
        self.target = []
        
        if self.reward != 0:
            #print('hey, the reward is this {}, so triggering reward steps'.format(self.reward))
            if self.reward == -2:
                h = game.history[-1]
                m = self.memory[-2][0]
                
                self.debug = m
                move = m[-1]
                
                items_short = m[:4]
                self.debug1 = items_short
                k = values.index(move)
                #value in array that corresponds to the move taken  
                items_short[k] = items_short[k]*self.reward
                self.target.append(items_short)
                
                self.X = agent.tokenize_board(game.history[-1]).reshape(1,-1)
                self.model.fit(np.array(self.X),np.array(self.target))
                print ('trained on invalid move')
                reward = 0 
                self.debug2 = 0
                return
                
            
            elif len(game.history) <= self.reward_depth:
                h = game.history[:-1]
                m = self.memory[-(len(game.history)-1):] #indexes into the last last point in memory attached to the current game.
                return # do nothing here to avoid unsolved index crash in items_short[0][:4]
                self.debug2 = 1
            else:
                h = game.history[-(1+self.reward_depth):-1] #index into these arrays from the back, up to a height of however far the depth is
                m = self.memory[-(1+self.reward_depth):] # memory is np array, game.history is list
                self.debug2 = 2
            #get moves 
            
            
            
           
            
            for items in m:  ##make mem shotened to the length you want, currently its five. 
                self.debug1 = m
                 #hacky nonsense way to check for this because it seems like the way I'm hadling these tensors is casting them into higher and lower order tensors
                #self.debug = items
                move = items[0][-1]

                items_short = items[0][:4]
                k = values.index(move)
                #value in array that corresponds to the move taken

                if self.reward*discount > 1:
                    items_short[k] = items_short[k]*self.reward*discount
                elif self.reward < 0:
                    items_short[k] = items_short[k]*self.reward*discount
                else:
                    items_short[k] = items_short[k]*1.05
                self.target.append(items_short)
                discount -=(1/self.reward_depth)
            self.X =[self.tokenize_board(i) for i in h]
            print('reward: {}'.format(self.reward))
            self.model.fit(np.array(self.X),np.array(self.target))
            self.reward = 0
         
        
        
    def neural_net(self,):
        '''
        a method which creates a neural net for the agent
        Returns:
        model, a tensorflow nn
        '''

        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim = 256 ))
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu') )
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(4, activation='softmax'))
        opt = Adam(self.learning_rate)
        
        model.compile(loss='mse', optimizer=opt)

        
        return model
