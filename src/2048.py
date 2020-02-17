import numpy as np

def add_tile(board):
    new_tile = random.choices([2,4], weights=[.8,.2])
    empties = np.where(board ==0)
    if type(empties) ==tuple:
        print('Game Over!')
        return 1
    empties= [(i,j) for i, j in zip(empties[0],empties[1])]
    tile = random.choice(empties)
    board[tile[0],tile[1]]=new_tile[0]
    return 0
    
def update_board(board):
    new_board=[]
    for row in board:
    
        if row[0] == row[1] and row[2] == row[3]:
            X = [row[0]+row[1], row[2]+row[3],0,0]
            new_row=[i for i in X if i != 0]
            while len(new_row)< 4:
                new_row.append(0)
        
        
        elif row[0]== row[1]:
            X = [row[0]+row[1],row[2],row[3]]
            new_row=[i for i in X if i != 0]
            while len(new_row)< 4:
                new_row.append(0)
          
        elif row[1] == row[2]:
            X = [row[0],row[2]+row[1],row[3]]
            new_row=[i for i in X if i != 0]
            while len(new_row)< 4:
                 new_row.append(0)      
        
        elif row[2] == row[3]:
            X = [row[0],row[1],row[2]+row[3]]
            new_row=[i for i in X if i != 0]
            while len(new_row)< 4:
                 new_row.append(0)
                    
        else:
            new_row = row
        new_board.append(new_row)
    return np.array(new_board)
