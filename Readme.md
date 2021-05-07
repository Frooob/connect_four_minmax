# Connect 4 against the minmax algorithm

In this project, you'll find a python implementation of the connect 4 algorithm, that you can play against an AI that is using a minmax algorithm. 
Note that the language of the sourcecode and the game hints is mostly german. 
Credits also go to Waldemar Schmidt, who collaborated with me in this project. 

## Usage
First install the requirementstxt
    
    $ pip install -r requirements.txt

### Singleplayer 
To play against the AI, simply run the script `python play_single.py`. 
You can edit the script to change the difficulty (number of turns the minmax algorithm evaluates in advance), the heuristic that the algorithm is using (you can chose between "optimum", "X" and "random") and also if the algorithm plays deterministically or makes a random weighted choice between the best possible options. 

#### Example:


### Multiplayer
To play against another person, simply run the script `python multiplayer.py`.