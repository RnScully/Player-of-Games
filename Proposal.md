Proposal


- ## What this project is trying to Achieve:
I'm trying to build a fast and light bot that plays 2048

The problem space: the game 2048:
2048 is a tile sliding game played on a 4 x 4 grid. The game starts with a board containing one tile, either a 2 or a 4. Every turn, the player may chose a direction to merge tiles. A merge can happen up, down, left, or right. In the merge, 0s are ignored, and tiles of same value combine into tiles worth twice their value. For example, a right merge:

0  0 2 2    0  0 0 4
8  0 8 0    0  0 0 16
4  4 4 4 -> 0  0 8 8
16 8 4 0    0 16 8 4

the top row, the twos slide to the right and form a 4. the next row, the tiles ignore 8s, slide to the right, and merge into a 16, the third row, the rightmost 4s become an 8, and the leftmost tiles become an 8, all tiles slide to the right. The fourth row, all tiles slide to the right, filling the space the 8 left empty. 

After the execution of the player's move, the game adds a 2 or a 4, 80% of the time a 2, to a random 0 tile. If there are no 0 tiles, the game ends. 

The titular goal of the game is to build a tile valued 2048, though a human can acheive this fairly regularly after a little practice, and the game continues. Scoring is calculated based upon the value of the merges in a running sum. Combining a 2 with a 2 gives you 4 points, a 4 and a 4 8 points, thus, to get to 16, you'll have made at least 32 points, but likely more since a player is likely to combine, intenonally or unavoidably, more tiles than what they absolutely need to combine. 


My objective is to create a fast player of 2048. One that scores well, above 60k points (acheiving a 2048 tile at least 3 times), and that does it, after training, in a computationally inexpensive way as a proof of concept for a system that trains better than route-following AI without human weight-setting.  


- ## This problem has been solved before with N-Tuple-Networks and Deep convolutional networks
State of the art deep convolutional Neural Nets can acheive scores on this game of 400,000 points, per Kondo and Matsuzaki 2019, and n-tuple-networks score an average scoreof 609,104 when given a max search time of one second. These exhaustive solutions are computationaly expensive, and not suitable for building fast players, furthermore, the N-Tuple solutions involved human guided initial weighting. 


- ## What is new about my Approach:
Search times of one second are human watchable, but not reasonable for instances where many agents are running, like multiplayer simulations, nor for game spaces with real-time interactions. This project has at its goal not mastery of the game space, but a learner who can perform like a moderately but not extremely skilled human very quickly. Nueral Net evolution can provide a robot that can react in real time, or in the search space of 2048, very quickly. 

- ## Work will be presented:
As a webapp allowing the viewer to view the AI playing throuhg the game, enter specific states and see the AI overcome or stumble on them, in a markdown writeup of the project, and a talk-track of the work.

- ## Data Sources, Storage, and Format:
The data source will be the states of play of the 4x4 2048 games, represented as a 1x16 vector, the storage will be done over a mongodb install, and the best model will be pickle-able. The AI will start with 

- ## Who cares: 
Video gaming is a 120 Billion dollar industry, and one of the biggest problems in games is player matchmaking and simulated opponent challeng feeling off. Furthermore, as game simulations become more realistic, computer agents that can operate in them approximate computer agents that can operate in the real world. There's significant overlap between a robot that can drive a mario-cart and a robot that drives (though the stakes for going off a lane in mario-care are much lower). Specifically to the case of making a fast player of 2048, that's a proof of concept for a wider set of possibilities. 

- ## Potential Problems:
Challenges: of the four possible moves, there one or more of them can be infeasible due to the game state, per Amarj and Dediu 2017. Maintaining low computational complexity while tying in enough complexity to perferm decently on the task also a problem. 



- ## Next thing I need to work on
I'm about halfway through creation of the game environment that the AI will be able to play in. After that is complete, the untrained neural net needs to be built. It will then propigate the data with game-states, and be able to learn from its own "best" failures to progress. 



Kondo and Matsuzaki 2019
https://www.jstage.jst.go.jp/article/ipsjjip/27/0/27_340/_pdf/-char/en


Amarj and Dediu 2017  
http://www.mit.edu/~amarj/files/2048.pdf found score-targeting to be inneficient way of incentivising growth

Evolving neural network to play game 2048
Boris and Sukovic
https://www.researchgate.net/publication/312569193_Evolving_neural_network_to_play_game_2048


https://towardsdatascience.com/how-to-teach-an-ai-to-play-games-deep-reinforcement-learning-28f9b920440a

https://towardsdatascience.com/neat-an-awesome-approach-to-neuroevolution-3eca5cc7930f





