import numpy as np
import random
import time, sys
from IPython.display import clear_output

class Game2048():
    def __init__(self, headless = False, ai = False, strict = False):
        
        self.board = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.add_tile() #initilaizes the board with a tile added. 
        self.score = 0
        self.current_move = None
        self.game_over = False
        self.history = []
        self.headless = headless
        self.ai = ai
        self.valid_moves = [True, True, True, True]
        self.strict = strict

    def get_move(self,ai_move=None):
        """
        Sets a players input to self.current_move.
        """
        if self.ai == False:
            print('4:slide left, 8:slide up, 6: slide right, 2:slide down, /n q: quit')
            self.current_move = input()
        else:
            self.current_move = ai_move


    def show_board(self):
        '''
        displays the current gameboard. ought work in-place to make game interactive without re-calling cells, but...that's not yet working. 
        '''

        #clear_output(wait = True)
        print(self.board)

    

    def add_tile(self):
        ''' adds a new tile either a 2 or 4 with 80% chance of a 2 in a random empty tile. 
        if game board is full, add_tile will call endgame handling methods. 
        '''
        empty_tiles = np.where(self.board ==0)
        if len(empty_tiles[0]) == 0 :
            self.game_over = True
            if self.headless == False:
                print('Game Over!')
            
            return(self.score)

        new_tile = random.choices([2,4], weights=[.8,.2])
        empty_idx= [(i,j) for i, j in zip(empty_tiles[0],empty_tiles[1])]
        tile = random.choice(empty_idx)
        self.board[tile[0],tile[1]]=new_tile[0]

    def turn_handling(self):
        '''
        method which handles the board's side of the turn. records the game state and adds random tiles
        '''
        self.add_tile()
        self.history.append(self.board.ravel())
        if self.headless == False:
            self.show_board()

    def slide_left(self)        :
        ''' slides all tiles to the left, ignoring 0s and merging all duplicates, records score as the sum of all merges. 
        +++++++++++
        Attributes:
            self.board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            Nothing
        +++++++++++
        Updates
        self.score (int): sum of all merges done in the method added to self.score
        self.board (np.array): current 4x4 boardstate 
        '''
        new_board = []
        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board]
        inner_score = []
        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[row[0],0,0,0]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[row[0]+row[1],0,0,0]
                    inner_score.append(row[0]+row[1])
                else:
                    new_row = [row[0],row[1],0,0]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [row[0]+row[1],row[2],0,0]
                    inner_score.append(row[0]+row[1])
                elif row[1]==row[2]:
                    new_row = [row[0],row[1]+row[2],0,0]
                    inner_score.append(row[1]+row[2])
                else:
                    new_row = [row[0],row[1],row[2],0]

            else: #other case len is 4
                    if row[0] == row[1] and row[2] == row[3]:
                        new_row = [row[0]+row[1], row[2]+row[3],0,0]
                        inner_score.append(row[0]+row[1])
                        inner_score.append(row[2]+row[3])     
                    elif row[0]== row[1]:
                        new_row = [row[0]+row[1],row[2],row[3],0]
                        inner_score.append(row[0]+row[1])
                    elif row[1] == row[2]:
                        new_row = [row[0],row[1]+row[2],row[3],0]
                        inner_score.append(row[1]+row[2])
                    elif row[2] == row[3]:
                        new_row = [row[0],row[1],row[2]+row[3],0]
                        inner_score.append(row[2]+row[3])
                    else:
                        new_row = row

            new_board.append(new_row)
        self.score += sum(inner_score)    
        return np.array(new_board)



    def slide_right(self):
        ''' slides all tiles to the right, ignoring 0s and merging all duplicates, records score as the sum of all merges. 
        +++++++++++
        Attributes:
            self.board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            Nothing
        +++++++++++
        Updates
        self.score (int): sum of all merges done in the method added to self.score
        self.board (np.array): current 4x4 boardstate 
        '''
        new_board=[]
        inner_score =[]
        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board]
        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[0,0,0,row[0]]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[0,0,0,row[0]+row[1]]
                    inner_score.append(row[0]+row[1])                    
                else:
                    new_row = [0,0,row[0],row[1]]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [0,0, row[0]+row[1],row[2]]
                    inner_score.append(row[0]+row[1])
                elif row[1]==row[2]:
                    new_row = [0,0,row[0],row[1]+row[2]]
                    inner_score.append(row[1]+row[2])
                    
                else:
                    new_row = [0,row[0],row[1],row[2]]

            else: #other case len is 4
                    if row[0] == row[1] and row[2] == row[3]:
                        new_row = [0,0, row[0]+row[1], row[2]+row[3]]    
                        inner_score.append(row[0]+row[1])
                        inner_score.append(row[2]+row[3])  
                    elif row[0]== row[1]:
                        new_row = [0,row[0]+row[1],row[2],row[3]]
                        inner_score.append(row[0]+row[1])
                    elif row[1] == row[2]:
                        new_row = [0,row[0],row[1]+row[2],row[3]]
                        inner_score.append(row[1]+row[2])
                    elif row[2] == row[3]:
                        new_row = [0,row[0],row[1],row[2]+row[3]]
                        inner_score.append(row[2]+row[3])
                    else:
                        new_row = row

            new_board.append(new_row)
        self.score += sum(inner_score)    
        return np.array(new_board)


    def slide_down(self):
        ''' slides all tiles down, ignoring 0s and merging all duplicates, records score as the sum of all merges. 
        +++++++++++
        Attributes:
            self.board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            Nothing
        +++++++++++
        Updates
        self.score (int): sum of all merges done in the method added to self.score
        self.board (np.array): current 4x4 boardstate 
        '''
        new_board=[]
        inner_score =[]
        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board.T]
        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[0,0,0,row[0]]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[0,0,0,row[0]+row[1]]
                    inner_score.append(row[0]+row[1])
                else:
                    new_row = [0,0,row[0],row[1]]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [0,0, row[0]+row[1],row[2]]
                    inner_score.append(row[0]+row[1])
                elif row[1]==row[2]:
                    new_row = [0,0,row[0],row[1]+row[2]]
                    inner_score.append(row[1]+row[2])
                else:
                    new_row = [0,row[0],row[1],row[2]]

            else: #other case len is 4
                    if row[0] == row[1] and row[2] == row[3]:
                        new_row = [0,0, row[0]+row[1], row[2]+row[3]]  
                        inner_score.append(row[0]+row[1])
                        inner_score.append(row[2]+row[3])    
                    elif row[2] == row[3]:
                        new_row = [0,row[0],row[1],row[2]+row[3]]
                        inner_score.append(row[2]+row[3])
                    elif row[0]== row[1]:
                        new_row = [0,row[0]+row[1],row[2],row[3]]
                        inner_score.append(row[0]+row[1])
                    elif row[1] == row[2]:
                        new_row = [0,row[0],row[1]+row[2],row[3]]
                        inner_score.append(row[1]+row[2])

                    else:
                        new_row = row

            new_board.append(new_row)
        self.score += sum(inner_score)    
        return np.array(new_board).T


    def slide_up(self):
        ''' slides all tiles to the right, ignoring 0s and merging all duplicates, records score as the sum of all merges. 
        +++++++++++
        Attributes:
            self.board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            Nothing
        +++++++++++
        Updates
        self.score (int): sum of all merges done in the method added to self.score
        self.board (np.array): current 4x4 boardstate 
        '''
        new_board=[]
        inner_score =[]
        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board.T]
        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[row[0],0,0,0]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[row[0]+row[1],0,0,0]
                    inner_score.append(row[0]+row[1])
                else:
                    new_row = [row[0],row[1],0,0]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [row[0]+row[1],row[2],0,0]
                    inner_score.append(row[0]+row[1])
                elif row[1]==row[2]:
                    new_row = [row[0],row[1]+row[2],0,0]
                    inner_score.append(row[1]+row[2])
                else:
                    new_row = [row[0],row[1],row[2],0]

            else: #other case len is 4
                if row[0] == row[1] and row[2] == row[3]:
                    new_row = [row[0]+row[1], row[2]+row[3],0,0]  
                    inner_score.append(row[0]+row[1])
                    inner_score.append(row[2]+row[3])   

                elif row[0]== row[1]:
                    new_row = [row[0]+row[1],row[2],row[3],0]
                    inner_score.append(row[0]+row[1])
                elif row[1] == row[2]:
                    new_row = [row[0],row[1]+row[2],row[3],0]
                    inner_score.append(row[1]+row[2])
                elif row[2] == row[3]:
                    new_row = [row[0],row[1],row[2]+row[3],0]
                    inner_score.append(row[2]+row[3])

                else:
                    new_row = row
            new_board.append(new_row)
        self.score += sum(inner_score)
        return np.array(new_board).T

    def game_step(self):
        '''
        a method which will step through one turn of 2048's gameplay loop, made to give the AI easier handles. 
        Attributes
        strict (bool) : True settings will end game if an invalid move has been made
        Returns 
        (-1): if move is invalid, game step will pass out a -1. 

        Updates
        self.valid_moves
        self.game_over
        self.current_move

        Calls:
        slide_right(), slide_left(), up() and down() to make moves
        turn_handling() to progress the turn. 
        '''
        
        completed_move = False 
        
        #Ai calls move and cannot give anything but 2, 4, 6, and 8
        #indexes of output in AI handler:
        # down, left, right, up
        # 2       4     6     8
        #[0]     [1]   [2]    [3]
        while completed_move == False:
            if self.valid_moves == [False, False, False, False]:
                break
            
            if self.current_move == 6:
                is_valid = self.slide_right()
                if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                    
                    self.board = is_valid
                    completed_move = True
                else:
                    if self.strict == True:
                        self.game_over = True
                    self.current_move = None
                    self.valid_moves[2] = False #updates the valid moves to tell the AI it can't keep retrying that. 
                    return -1
                        
            elif self.current_move == 8:
                is_valid = self.slide_up()
                if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                    self.board = is_valid
                    completed_move = True
                else:
                    if self.strict == True:
                        self.game_over = True
                    self.current_move = None
                    self.valid_moves[3] = False #updates the valid moves to tell the AI it can't keep retrying that. 
                    return -1
                        
            elif self.current_move == 4:
                is_valid = self.slide_left()
                if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                    self.board = is_valid
                    completed_move = True
                else:
                    if self.strict == True:
                        self.game_over = True
                    self.current_move = None
                    self.valid_moves[1] = False #updates the valid moves to tell the AI it can't keep retrying that. 
                    return -1

            elif self.current_move == 2:
                is_valid = self.slide_down()
                if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                    self.board = is_valid
                    completed_move = True
                else:
                    if self.strict == True:
                        self.game_over = True
                    self.current_move = None
                    self.valid_moves[0] = False #updates the valid moves to tell the AI it can't keep retrying that. 
                    return -1
            
            else:
                print('something has gone wrong in game_step!')
                print(self.current_move)
            self.valid_moves = [True, True, True, True]
            self.turn_handling()



    def game_loop(self):
        '''
        a method which should operate 2048's gameplay loop. (show board, wait for move, update board, add new tile, loop)
        '''
        while self.game_over == False:
            
            
            if self.current_move == 'q':
                break
            self.current_move = None
            while self.current_move == None:
                self.get_move()
                if self.current_move == "q":
                    break

                elif self.current_move == '6' or self.current_move.lower() =='a':
                    is_valid = self.slide_right()
                    if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                        self.board = is_valid
                    else:
                        print('please use a valid move')
                        self.current_move = None
                        continue


                elif self.current_move == '8' or self.current_move.lower() =='w':
                    is_valid = self.slide_up()
                    if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                        self.board = is_valid
                    else:
                        print('please use a valid move')
                        self.current_move = None
                        continue

                elif self.current_move == '4' or self.current_move.lower() =='d':
                    is_valid = self.slide_left()
                    if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                        self.board = is_valid
                    else:
                        print('please use a valid move')
                        self.current_move = None
                        continue
                    
                elif self.current_move == '2' or self.current_move.lower() =='s':
                    is_valid = self.slide_down()
                    if np.array_equal(self.board, is_valid) == False: #checks to see if the move entered is a valid move (you cannot make moves that don't change the board)
                        self.board = is_valid
                    else:
                        print('please use a valid move')
                        self.current_move = None
                        continue
                    
                else:
                    print('please use a valid move')
                    self.current_move = None
                    continue

                self.turn_handling()

if __name__ == "__main__":
    g = Game2048()
    
    g.game_loop()
    print(g.score)    
    