import numpy as np
import random
import time, sys
from IPython.display import clear_output

class Game2048():
    def __init__(self):
        
        self.board = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
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
        if len(empty_tiles) == 0 :
            print('Game Over!')

        new_tile = random.choices([2,4], weights=[.8,.2])
        empty_idx= [(i,j) for i, j in zip(empty_tiles[0],empty_tiles[1])]
        tile = random.choice(empty_idx)
        self.board[tile[0],tile[1]]=new_tile[0]


    def slide_left(self):
        ''' slides all tiles to the left, ignoring 0s and merging all duplicates
        +++++++++++
        Attributes:
            board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            new_board (np.array): the updated 4x4 board after a player move
        '''
        new_board = []
        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board]

        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[row[0],0,0,0]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[row[0]+row[1],0,0,0]
                else:
                    new_row = [row[0],row[1],0,0]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [row[0]+row[1],row[2],0,0]
                elif row[1]==row[2]:
                    new_row = [row[0],row[1]+row[2],0,0]
                else:
                    new_row = [row[0],row[1],row[2],0]

            else: #other case len is 4
                    if row[0] == row[1] and row[2] == row[3]:
                        new_row = [row[0]+row[1], row[2]+row[3],0,0]      
                    elif row[0]== row[1]:
                        new_row = [row[0]+row[1],row[2],row[3],0]
                    elif row[1] == row[2]:
                        new_row = [row[0],row[2]+row[1],row[3],0]
                    elif row[2] == row[3]:
                        new_row = [row[0],row[1],row[2]+row[3],0]
                    else:
                        new_row = row

            new_board.append(new_row)
        self.board = np.array(new_board)



    def slide_right(self):
        ''' slides all tiles to the right, ignoring 0s and merging all duplicates
        +++++++++++
        Attributes:
            board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            new_board (np.array): the updated 4x4 board after a player move
        '''
        new_board=[]
        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board]
        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[0,0,0,row[0]]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[0,0,0,row[0]+row[1]]
                else:
                    new_row = [0,0,row[0],row[1]]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [0,0, row[0]+row[1],row[2]]
                elif row[1]==row[2]:
                    new_row = [0,0,row[0],row[1]+row[2]]
                else:
                    new_row = [0,row[0],row[1],row[2]]

            else: #other case len is 4
                    if row[0] == row[1] and row[2] == row[3]:
                        new_row = [0,0, row[0]+row[1], row[2]+row[3]]      
                    elif row[0]== row[1]:
                        new_row = [0,row[0]+row[1],row[2],row[3]]
                    elif row[1] == row[2]:
                        new_row = [0,row[0],row[2]+row[1],row[3]]
                    elif row[2] == row[3]:
                        new_row = [0,row[0],row[1],row[2]+row[3]]
                    else:
                        new_row = row

            new_board.append(new_row)
        self.board = np.array(new_board)


    def slide_down(self):
        ''' slides all tiles down, ignoring 0s and merging all duplicates
        +++++++++++
        Attributes:
            board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            new_board (np.array): the updated 4x4 board after a player move
        '''
        new_board=[]
        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board.T]
        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[0,0,0,row[0]]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[0,0,0,row[0]+row[1]]
                else:
                    new_row = [0,0,row[0],row[1]]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [0,0, row[0]+row[1],row[2]]
                elif row[1]==row[2]:
                    new_row = [0,0,row[0],row[1]+row[2]]
                else:
                    new_row = [0,row[0],row[1],row[2]]

            else: #other case len is 4
                    if row[0] == row[1] and row[2] == row[3]:
                        new_row = [0,0, row[0]+row[1], row[2]+row[3]]      
                    elif row[2] == row[3]:
                        new_row = [0,row[0],row[1],row[2]+row[3]]
                    elif row[0]== row[1]:
                        new_row = [0,row[0]+row[1],row[2],row[3]]
                    elif row[1] == row[2]:
                        new_row = [0,row[0],row[1]+row[2],row[3]]

                    else:
                        new_row = row

            new_board.append(new_row)
        self.board =  np.array(new_board).T


    def slide_up(self):
        ''' slides all tiles up, ignoring 0s and merging all duplicates
        +++++++++++
        Attributes:
            board (np.array): the 4x4 gameboard
        +++++++++++
        Returns:
            new_board (np.array): the updated 4x4 board after a player move
        '''
        new_board=[]

        board_no_zeros=[[i for i in rows if i != 0] for rows in self.board.T]
        for row in board_no_zeros:
            if len(row) == 0:
                new_row =[0,0,0,0]
            elif len(row) ==1:
                new_row =[row[0],0,0,0]
            elif len(row) ==2:
                if row[0]==row[1]:
                    new_row =[row[0]+row[1],0,0,0]
                else:
                    new_row = [row[0],row[1],0,0]

            elif len(row)==3:
                if row[0]==row[1]:
                    new_row = [row[0]+row[1],row[2],0,0]
                elif row[1]==row[2]:
                    new_row = [row[0],row[1]+row[2],0,0]
                else:
                    new_row = [row[0],row[1],row[2],0]

            else: #other case len is 4
                if row[0] == row[1] and row[2] == row[3]:
                    new_row = [row[0]+row[1], row[2]+row[3],0,0]      

                elif row[0]== row[1]:
                    new_row = [row[0]+row[1],row[2],row[3],0]
                elif row[1] == row[2]:
                    new_row = [row[0],row[1]+row[2],row[3],0]
                elif row[2] == row[3]:
                    new_row = [row[0],row[1],row[2]+row[3],0]

                else:
                    new_row = row

            new_board.append(new_row)

        self.board =  np.array(new_board).T