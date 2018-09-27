## Board state
The player specific board state needs to encode which phase the player and the opponend is in (phase 1 placing, phase 2 moving, phase 3 flying)
Those values can be encoded into the center values of the board (the phases could also be derived from the prisoner count and tile count of each player, but this would make it more complicated to learn)
### 2
- players and opponends stone positions
- one hot encoding for each players phase

### 3
3x3x11
first three planes express the board with

- const plane "select from"-state
- const plane "select to"-state
- const plane "imprisonopponent"-state
- const plane "prisoner count"
- const plane "opponent prisoner count"
- const plane "identical states"
- const plane "turns without mill" (50-turns until draw)
- const plane "player with turn"

The board gets inverted after every turn switch, so that the model can train on a agnostic view

## Action space
The actions need to be independend of the phase, therefore they will be encoded with the triple
(from, to, stone to take). Stone to remove if a mill has been closed (stoneid 0..8)

in phase 1 from will be ignored (or we add a dummy value to encode that the stone is taken from the reservour)
in phase 2 from needs to contain a stone and to could be encoded by the 4 directions, but for flying we need actually all board positions!!!

## General ideas
- the gamestate should be encoded inside the Board class, so if the model will be changed, it has to be changed only on one position (the Game class should mostly use the Board class and work little on the Tensor)
- if its only the board encoding, one can work on the tensor directly, as it is faster than using the class