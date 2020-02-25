## Welcome to Models!

This is a historical markdown of models at steps. 

# Models
initial training feature: score/10,000
### Monday_Mk1 
The first model to be spit out of the evolution, before any edits were made. No matter what, she says slide up. 

This was when I realized I needed to build out systems for handling invalid moves (you can't move up when nothing else will slide up, for example) and built the method which takes the AI's suggestions and filters them for allowed options. 

After that, the model says slide if she can, if not she slides right, if not, left, if not down. (she never actually slides down, she loses before that happens)


As a rudimentary strategy, this does rather well, as it works to keep her tiles in a gradient and gets the biggest ones in a corner. 

### Tuesday_Mk1

I allowed the model to grow recurrent nodes in this evolution, and changed the  training feature to reward the bot every time a move resulted in an additional open tile. (fitness became -1/(the sum of number of empty tiles created in a game)+1

This model has learned to keep its biggest tiles on top, and tends to switch between sliding left and right as a second priority. (does this switching on its own, not by being told that moves are illegal). In its worst playthroughs, it does about as well as monday_mk1, but in its best, It does much better. 
