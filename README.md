Proposal


- ## What this project is trying to Achieve:
I want to create a pipeline for training lightweight players of games, as a proof of concept that will eventually create more interesting oppoents in video game situations. 

The problem space: the game 2048:
2048 is a tile sliding game played on a 4 x 4 grid where square tiles hold powers of two, and two same tiles combine to create a tile twice that tiles value. The titular goal of the game is to combine tiles untill you have a tile valued at 2048, but the game goes on much further than that. It is scored by a sum of the value of all combinations done to reach the current game state. 

My objective is to create a fast player of 2048. One that scores well, above 60k points (acheiving a 2048 tile at least 3 times), and that does it, after training, in a computationally inexpensive way as a proof of concept for a system that trains better than route-following AI without human tinkering. 


- ## This problem has been solved before with N-Tuple-Networks and Deep convolutional networks
State of the art deep convolutional Neural Nets can acheive scores on this game of 400,000 points, per Kondo and Matsuzaki 2019, and n-tuple-networks score an average scoreof 609,104 when given a max search time of one second. These exhaustive solutions are computationaly expensive, and not suitable for building 


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





