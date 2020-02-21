import numpy as np
import random
import time, sys
from IPython.display import clear_output

class Game2048(headless = False):
    def __init__(self):
        
        self.board = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.score = 0
        self.current_move = None
        self.game_over = False
        self.history = []
        self.headless = headless

    def get_move(self):
        """
        Sets a players input to self.current_move.
        """
        print('4:slide left, 8:slide up, 6: slide right, 2:slide down, /n q: quit')
        self.current_move = input()


    def show_board(self):
        '''
        displays the current gameboard. ought work in-place to make game interactive without re-calling cells, but...that's not yet working. 
        '''

        #clear_output(wait = True)
        print(self.board)

    def turn_handling(self)
        add_tile()
        self.history.append(self.board.ravel())

    def add_tile(self):
        ''' adds a new tile either a 2 or 4 with 80% chance of a 2 in a random empty tile. 
        if game board is full, add_tile will call endgame handling methods. 
        '''
        empty_tiles = np.where(self.board ==0)
        if len(empty_tiles) == 0 :
            self.game_over = True
            print('Game Over!')

        new_tile = random.choices([2,4], weights=[.8,.2])
        empty_idx= [(i,j) for i, j in zip(empty_tiles[0],empty_tiles[1])]
        tile = random.choice(empty_idx)
        self.board[tile[0],tile[1]]=new_tile[0]


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
        self.board = np.array(new_board)



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
        self.board = np.array(new_board)


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
        self.board =  np.array(new_board).T


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
        self.board =  np.array(new_board).T

    def game_loop(self):
        while self.game_over == False:
            self.turn_handling()
            self.show_board()
            if self.current_move == 'q':
                break
            self.current_move = None
            while self.current_move == None:
                self.get_move()
                if self.current_move == "q":
                    break
                elif self.current_move == '6' or self.current_move.lower() =='a':
                    self.slide_right()
                elif self.current_move == '8' or self.current_move.lower() =='w':
                    self.slide_up()
                elif self.current_move == '4' or self.current_move.lower() =='d':
                    self.slide_left()
                elif self.current_move == '2' or self.current_move.lower() =='s':
                    self.slide_down()
                else:
                    print('please use a valid move')
                    self.current_move = None
                    continue

if __name__ == "__main__":
    g = Game2048()
    
    g.game_loop()
    print(g.score)    
    