#Player of Games
a 2048 N



/home/robert/Player-of-Games/src/2048.py

## The problem space: the game 2048:

2048 is a tile sliding game played on a 4 x 4 grid. The game starts with a board containing one tile, either a 2 or a 4. Every turn, the player may chose a direction to merge tiles. A merge can happen up, down, left, or right. In the merge, 0s are ignored, and tiles of same value combine into tiles worth twice their value. For example, a right merge:

0  0 2 2    0  0 0 4
8  0 8 0    0  0 0 16
4  4 4 4 -> 0  0 8 8
16 8 4 0    0 16 8 4

the top row, the twos slide to the right and form a 4. the next row, the tiles ignore 8s, slide to the right, and merge into a 16, the third row, the rightmost 4s become an 8, and the leftmost tiles become an 8, all tiles slide to the right. The fourth row, all tiles slide to the right, filling the space the 8 left empty. 

After the execution of the player's move, the game adds a 2 or a 4, 80% of the time a 2, to a random 0 tile. If there are no 0 tiles, the game ends. 

The titular goal of the game is to build a tile valued 2048, though a human can acheive this fairly regularly after a little practice, and the game continues. Scoring is calculated based upon the value of the merges in a running sum. Combining a 2 with a 2 gives you 4 points, a 4 and a 4 8 points, thus, to get to 16, you'll have made at least 32 points, but likely more since a player is likely to combine, intenonally or unavoidably, more tiles than what they absolutely need to combine. 


# Process

- Build a representation of the game in simple python that can run headlessly
