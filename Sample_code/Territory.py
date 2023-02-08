"""

AlphaGear: Territory Class

@authors: David Chataway, Alec Meade 

"""

###############################################################################

from sys import argv
from random import randrange

class territory():
    # The class "territory" contains the information about the territory and the following methods:
    #   - place (adds troops to the territory)
    #   - fortify (removes troops from the territory and adds it to another specified territory)  
    #   - attack

    def __init__(self, name, continent, region, bordering_territories, owner):
        self.name = name
        self.troops = 0
        self.continent = continent
        self.region = region
        self.bordering_territories = bordering_territories
        self.owner = owner

    # PLACE
    def place(self, amount):
        if amount <= 0:
            raise ValueError('Error: must fortify a positive amount.')
        else:
            self.troops = self.troops + amount
            print("Placed " + self.name + " with " + str(amount) + " to result in " + str(self.troops))

    # FORTIFY
    def fortify(self, amount, to_territory):
        # 'to_territory' must be an object
        if amount <= 0 or to_territory.name not in self.bordering_territories:
            raise ValueError('Error: must fortify a positive amount or fortify to a bordering territory.')
        elif (self.troops - amount)<1:
            raise ValueError('Error: there must be at least 1 troop remaining')
        else:
            self.troops = self.troops - amount
            print("\nFortified " + str(amount) + " troops from " + self.name + " to " + to_territory.name)
            to_territory.place(amount)

    # ATTACK
    def attack(self, amount, defending_territory):
        # note: 'defending_territory' must be an object
        
        if amount <= 0 or defending_territory.name not in self.bordering_territories:
            raise ValueError('Error: must attack with a positive amount or attack a bordering territory.')
        elif amount > self.troops:
            raise ValueError('Error: cannot attack with more troops than on the territory')
        else:
            # Logic for the attack, in part based on this https://github.com/attoPascal/risk-simulator/blob/master/risk.py
            print("\nAttacked with " + str(amount) + " troops from " + self.name + " to " + defending_territory.name)
            remaining_attacking_amount = amount
            
            # Create while loop until a) attackers hit 1 troop (out of selected attacking amount) or b) attackers win (defenders have no troops)
            while remaining_attacking_amount>1 and defending_territory.troops>0:
                # Simulate each roll
                # Generate the random die rolls for each side based on the number of troops
                attack_dice = attack_roll(remaining_attacking_amount)
                print("Attacker rolls: " + str(attack_dice))
                defense_dice = defend_roll(defending_territory.troops)
                print("Defender rolls: " + str(defense_dice))
                
                # Determine the outcomes of the turn (while loop contains a turn) and update troop amounts
                while len(attack_dice) > 0 and len(defense_dice) > 0:
                    # Since the arrays are sorted, you can compare positions index-wise
                    if attack_dice.pop(0) > defense_dice.pop(0):
                        defending_territory.troops -= 1
                    else:
                        self.troops -= 1
                        remaining_attacking_amount -= 1

            # After exiting the while loop (indicating a result occured), if the attacker wins, provide option of fortifying troops  
            if(remaining_attacking_amount>1):

                #change owner of the conquered territory
                defending_territory.owner = self.owner

                #mandatory transfer of troops
                defending_territory.troops = (remaining_attacking_amount-1)
                self.troops -= (remaining_attacking_amount-1)

                print(self.owner + " won the attack!\n" + self.name + " lost: " + str(amount-remaining_attacking_amount) + ", transfered: " + str(remaining_attacking_amount-1) + " troops and has " + str(self.troops -1) + " troops remaining available to be moved as well.")

                #optional transfer of troops
                # Requests an input to fortify and only breaks when the amount is acceptably 1 less than the attacking amount 
                if (self.troops) > 1:
                    fortify_amount = int(input('\nEnter the number of troops (less than or equal to ' + str(self.troops - 1) + ') that you would like to fortify into your captured territory: '))
                    self.fortify(fortify_amount, defending_territory)
                else:
                    pass
            
            else:
                print(defending_territory.owner + " won the attack!")

    # Display properties
    def __str__(self):
        return f"\nTerritory is: {self.name} \nThe region it is in is: {self.region}\nThe continent it is in is: {self.continent}\nOwner is: {self.owner}\nWith the following troops: {self.troops}\n"
    def __len__(self):
        return f"The number of troops in {self.owner} is: {self.troops}"

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

###############################################################################
########################       DEMO       #####################################
###############################################################################

def main():
    Quebec = territory("Quebec", "North America", "Eastern Canada", "Ontario","Player A")
    Quebec.place(8)
    print(Quebec)

    Ontario = territory("Ontario", "North America", "Eastern Canada", "Quebec","Player B")
    Ontario.place(3)
    print(Ontario)

    Quebec.attack(5,Ontario)

    #Ontario.fortify(1, Quebec)
    print(Quebec)
    print(Ontario)

if __name__ == '__main__':
    main()


