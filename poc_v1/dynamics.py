"""
Functions that define the dynamics of the AlphaGear POC v1

"""

###############################################################################

from sys import argv
from random import randrange

# The function will take in the state and action, and return the next state

# max number of actions = 5 x number of states
# attack up, down, left, right and do nothing

# no fortify step
# attack goes to completion

def action_state(state, action_num, num_actions):
    
    # from action_num, determine the territory [from , to, itself?]
    action_map = {}
    
    '''
    [0] [1] [2] [3] [4]
    [5] [6] [7] [8] [9]
    [10][11][12][13][14]
    [15][16][17][18][19]
    [20][21][22][23][24]
    '''
    
    dim = int((len(state))**0.5)

    for i in range(len(state)):
        # each territory has 5 actions: up, down, left, right, do nothing
        # if out of bounds, then it doesn't do anything
        action_map[i*dim] = [i, i-dim, False] if i-dim >= 0 else [i, i, True] # up
        action_map[i*dim + 1] = [i, i+dim, False] if i+dim < len(state) else [i, i, True] # down
        action_map[i*dim + 2] = [i, i-1, False] if (i)%dim != 0 else [i, i, True] # left
        action_map[i*dim + 3] = [i, i+1, False] if (i)%dim != (dim-1) else [i, i, True] # right
        action_map[i*dim + 4] = [i, i, True] # do nothing    
    assert len(action_map) == num_actions
    
    # conditions to see if you can proceed
    condition_ownership = state[action_map[action_num][0]] > 0  # if the territory is owned by you (i.e. it has positive troops)
    condition_enemy = state[action_map[action_num][1]] < 0      # if the territory you are attacking is not owned by you
    condition_valid = action_map[action_num][2] != True         # if the attack action is valid (i.e. not 'do nothing')
    
    # place troops
    if condition_ownership:
        state = place(state, action_map[action_num][0])
    
    # attack (if relevant)
    if condition_ownership and condition_enemy and condition_valid:
        state = attack(state, action_num, action_map)
    
    # return the state after the action is taken
    return state


# PLACE
def place(state, territory):
    # placing troops
    # check if they have a bonus group +2
    # +3 to attacking territory
    
    amount = 3
            
    # Provides +2 bonus if all territories in the corner 2x2 are owned
    # NOTE: assumes C type (left to right) ordering of the grid
    dim = int((len(state))**0.5)

    # top left corner
    if (state[0] > 0) & (state[0 + 1] > 0) & (state[dim] > 0) & (state[dim+1] > 0):
        amount += 2
    # top right corner
    if (state[dim-2] > 0) & (state[dim-1] > 0) & (state[2*dim-2] > 0) & (state[2*dim-1] > 0):
        amount += 2
    # bottom left corner
    if (state[(dim-2)*(dim)] > 0) & (state[(dim-2)*(dim)+1] > 0) & (state[(dim-1)*(dim)] > 0) & (state[(dim-1)*(dim)+1] > 0):
        amount += 2
    # bottom right corner
    if (state[(dim-1)*(dim)-2] > 0) & (state[(dim-1)*(dim)-1] > 0) & (state[(dim)*(dim)-2] > 0) & (state[(dim)*(dim)-1] > 0):
        amount += 2
        
    state[territory] += amount
    
    return state


# ATTACK
def attack(state, action_num, action_map):
    # Logic for the attack, in part based on this https://github.com/attoPascal/risk-simulator/blob/master/risk.py
    # Stipulations: a) it plays through the entire attack, b) you must move all troops
    
    attacking_territory = action_map[action_num][0]
    remaining_attacking_amount = state[attacking_territory]
    defending_territory = action_map[action_num][1]
    remaining_defending_amount = abs(state[defending_territory])
    
    if remaining_attacking_amount <= 0:
        raise ValueError('Error: must attack with a positive amount')
        
    # Create simulate rolls until a) attackers hit 1 troop (unable to attack further) or b) attackers win (defenders have no troops)
    while remaining_attacking_amount>1 and remaining_defending_amount>0:
        # Simulate each roll - generate the random die rolls for each side based on the number of troops
        attack_dice = attack_roll(remaining_attacking_amount)
        defense_dice = defend_roll(remaining_defending_amount)

        # Determine the outcomes of the turn (while loop contains a turn) and update troop amounts
        while len(attack_dice) > 0 and len(defense_dice) > 0:
            # Since the arrays are sorted, you can compare positions index-wise
            if attack_dice.pop(0) > defense_dice.pop(0):
                # attacker wins
                remaining_defending_amount -= 1
            else:
                # attacker loses
                remaining_attacking_amount -= 1

    # After exiting the while loop (indicating a result occured)
    # attacker wins
    if(remaining_attacking_amount>1):
        #mandatory transfer of troops
        state[attacking_territory] = 1
        state[defending_territory] = (remaining_attacking_amount-1)

    else:
        # defender wins
        state[attacking_territory] = remaining_attacking_amount
        state[defending_territory] = (remaining_defending_amount)*-1 #correct for abs, opponent has '-' troops 
        
    return state


###############################################################################
def roll_dice():
    return randrange(1,7)

def attack_roll(units):
    # roll with 1 fewer dice than the number of units used
    if units > 3:
        cast = [roll_dice(), roll_dice(), roll_dice()]
    elif units == 3:
        cast = [roll_dice(), roll_dice()]
    elif units ==2:
        cast = [roll_dice()]
    else:
        raise ValueError('Error: cannot attack with 1 troop')

    return sorted(cast, reverse=True)

def defend_roll(units):
    if units >= 2:
        cast = [roll_dice(), roll_dice()]
    else:
        cast = [roll_dice()]

    return sorted(cast, reverse=True)



