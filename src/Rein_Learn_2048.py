

from g2048 import Game2048
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

class RL_Player():
    def __init__(self):
            self.reward = 0
            self.gamma = 0.9
            # self.dataframe = pd.DataFrame()
            self.short_memory = np.array([])
            self.agent_target = 1
            self.agent_predict = 0
            self.learning_rate = 0.0005
            self.model = self.neural_net()
            #self.model = self.network("weights.hdf5")
            self.epsilon = 0
            self.actual = []
            self.memory = []
            
    

    def ai_suggest_move(self, game):
        board = game.board.ravel
        pred= self.model.predict(game.board.ravel().reshape(1,-1))
        output = [i for i in pred[0]]
        
        values = [2, 4, 6, 8]
        valid_moves = game.valid_moves

        d = dict(zip(np.array(output)[valid_moves], np.array(values)[valid_moves]))
        if len(d.keys()) == 0:
            #print('ai_suggest_move is out of possible moves')
            game.game_over = True #game is over
            return  -1

        if game.strict == True:
            if max(output) in d.keys(): #if the suggested move is not in valid_moves, it wont be in d, and so this will be negative, and the game will end
                move = d.get(max(d.keys()))
                return move
            else:
                game.game_over = True #game is over because this setting does not allow invaild moves:
                return  -1
        else:
            move = d.get(max(d.keys())) #takes the next best move that is valid
            return move
    
    def give_reward(self):
        old_board = game.board()
        new_board = game.get_move(self.ai_suggest_move(self.model, game))
        
    def neural_net(self, weights = None):

        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape= (16,)  ))
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(4, activation='softmax'))
        opt = Adam(self.learning_rate)
        
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights
        return model
        
