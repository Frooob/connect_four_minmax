import numpy as np
import operator
import os
from termcolor import colored
import random 
from collections import defaultdict
from time import sleep
class VGW():
    def __init__(self):
        """ Das Spielbrett ist  6 x 7 Felder groß
            Es werden alle Züge in "letzter_zug" gespeichert
        """
        self.brett = np.zeros((6,7))
        # jeder Zug ist aufgebaut aus: (h, position, spieler)  // h entspricht der erstbesten, nicht belegten Höhe der Matrix
        self.baum = ""
        self.letzter_zug = [(0,0,0)]    

    def __str__(self):
        sl = "|"
        for line in self.brett:
            for value in line:
                if value == 0:
                    sl = sl+"   |"  
                elif value == 1:
                    sl = sl+" "+colored("x", "red")+" |"
                else:
                    sl = sl+" "+colored("o", "blue")+" |"
            sl = sl + "\n-----------------------------"
            sl = sl + "\n|"
        sl = sl [:-2]
        return sl

    def zug_erlaubt(self, position):
        return self.brett[0, position] == 0

    def zug(self, position, spieler = None):
        """ Spieler ist 1 oder 2.
            Position ist 0-6
        """
        if spieler == None:
            spieler = 1 if self.letzter_zug[-1][2] == 2 else 2
        # Höhe finden an der eingefügt werden muss
        if not self.zug_erlaubt(position): 
            print(f"Das Feld {position} ist schon voll, du kannst hier nicht legen.")
            return False

        h = np.argmax(self.brett[:,position] > 0)-1
        h = 5 if h <0 else h 
        self.brett[h,position]=spieler
        self.letzter_zug.append((h,position, spieler))
        return True
 
    def undo(self):
        """ führt das Brett in den Zustand vor dem aktuellen Zug zurück
        """
        h, position, _ = self.letzter_zug.pop()
        self.brett[h, position] = 0

    def get_current_state(self):
        a_tuple = (self.brett, self.letzter_zug)
        return a_tuple

    def set_to_state(self, statetuple):
        self.brett = np.array(statetuple[0])
        self.letzter_zug = statetuple[1]
        return
    
    def diagonale(self, h, position, spieler, dir):
        """ Überprüft für den letzten Zug ob der Spieler mit einer Diagonale gewonnen hat.
            dir muss entweder 1 oder 2 sein für beide möglichen Diagonalen
        """
        win = 0
        op = operator.add if dir == 1 else operator.sub
        for i in range(-3, 4): 
            x = position + i
            y = op(h,i)
            if y < 0 or y > 5:
                continue
            if x < 0 or x > 6:
                continue
            if self.brett[y, x] == spieler:
                win = win+1
                if win == 4:
                    return True
            else:
                win = 0
        return False
    
    def waagerecht(self, h, position, spieler):
        """ Überprüft für den letzten Zug ob der Spieler mit einer Waagerechten gewonnen hat.
        """
        x_min = 0 if position -3 < 0 else position -3
        x_max = 6 if position + 4 > 6 else position + 4
        win = 0
        waagerechte = self.brett[h, x_min : x_max]
        for p in waagerechte:
            if p == spieler:
                win = win +1
                if win == 4:
                    return True
            else:
                win = 0
        return False

    def gewinner_ermitteln(self):
        """ Da der letzte Zug bekannt ist muss nur vom letzten belegten
            Feld aus geguckt werden ob eine person das Spiel gewonnen hat.
        """
        if self.letzter_zug[-1] == (0,0,0):
            print("Es wurde noch kein Zug gemacht, es kann noch keinen Gewinner geben.")
        h, position, spieler = self.letzter_zug[-1]

        w = self.waagerecht(h, position, spieler)
        vu = self.brett[h:h+4, position]== spieler # vertikal unten
        d1 = self.diagonale(h, position, spieler, 1) # diagonale 1 
        d2 = self.diagonale(h, position, spieler, 2) # diagonale 2
        if any((w, np.sum(vu)==4,d1,d2)):
            return spieler
        else:
            return 0
    
    def multiplayer(self):
        spieler = 1
        os.system('clear')
        while True:
            
            print(self)
            sign = "x" if spieler == 1 else "o"
            color = "red" if spieler == 1 else "blue"
            user_input = input(f"spieler{spieler} ("+colored(sign, color)+") wo solls hingehen?")
            if user_input == "undo":
                self.undo()
                spieler = 1 if spieler == 2 else 2
                continue
            position = int(user_input)
            
            while not (self.zug(position, spieler)):
                position = int(input(f"spieler{spieler} wo solls hingehen?"))
            os.system('clear')
            spieler = 1 if spieler == 2 else 2
            if self.gewinner_ermitteln():
                print("Du hast gewonnen!")
                print(self)
                break
    
    def single_player(self, schwierigkeit, heuristik, **kwargs):
        os.system('clear')
        spieler = 1
        gegner = 2
        while True:
            print(self)
            user_input = input(f"spieler{spieler} ("+colored("x", "red")+") wo solls hingehen?")
            position = int(user_input)
            while not position in range(7):
                print("Diese Position ist nicht möglich. Bitte eine Zahl zwischen 0 und 6 eingeben.")
                user_input = input(f"spieler{spieler} ("+colored("x", "red")+") wo solls hingehen?")
                position = int(user_input)
            
            while not (self.zug(position, spieler)):
                position = int(input(f"spieler{spieler} wo solls hingehen?"))
            if self.gewinner_ermitteln():
                os.system('clear')
                print(self)
                print("SPIEL VORBEI. Spieler "+ str(self.gewinner_ermitteln()) + " hat gewonnen!")
                break
            # gegner wählt seinen Zug gemäß einer Heuristik
            gegner_zugnummer = self.minmax(schwierigkeit, gegner, heuristik, **kwargs) # Hier die gewünschte Heuristik übergeben  # schwierigkeit == tiefe
            os.system('clear')
            if gegner_zugnummer == 999:
                print("Unentschieden!")
                break
            self.zug(gegner_zugnummer)
            if self.gewinner_ermitteln():
                print(self)
                print("SPIEL VORBEI. Spieler "+ str(self.gewinner_ermitteln()) + " hat gewonnen!")
                break

    # waldemar
    def heuristik_zufall(self, spieler, *args):
        """ Eine Heuristik braucht die Informationen: 
        - das Spielfeld in dem der zu bewertende Zug schon ausgeführt wurde.
        - den Spieler der einen Zug machen will. (am Anfang der Rekursion)

        Die Heuristik bewertet dann diesen Zug in Hinblick darauf wie gut er für den Ursprungsspieler ist.

        Heuristik Zufall gibt aber einfach nur einen zufälligen Wert zurück.   
        """
        return random.randint(1,50)

    def heuristic_evaluate_optimum(self, spieler, *args):
        """ Bewertet ein Spielbrett aus Sicht des übergebenen Spielers
            verwendet das Bewertungsschema
            gibt als int eine Summe zurück

        """
        total = 0
        # Punkte für mittlere Spalte
        total += self.evaluate_middle(spieler)
        # Punkte für alle Reihen, Spalten und Diagonalen, die mehr als 4 Plätze umfassen
        all_lines = lines(self.brett)
        a_dictionary = eval_lines(all_lines)
        total += self.score_cases(a_dictionary, spieler)
    
        return total

    def score(self, argument):
        """ Nachfolgend das Bewertungsschema für die Optimum Heuristik
        """
        switcher = {
            'spieler_2' : 5 ,         #2er Folge
            'spieler_3' : 10 ,        #3er Folge
            'spieler_4' : 100000 ,    #4er folge
            'spieler_mid' : 2 ,       #mittlere Spalte

            'gegner_2' : -2 ,         #2er Folge Gegner
            'gegner_3' : -5 ,         #3er Folge Gegner
            'gegner_4' : -50000       #4er Folge Gegner
        }
        return switcher.get(argument)

    def score_cases(self, a_dictionary, spieler):
        """ Bewertet alle im Dictionary gesammelten Vorkommen gemäß der Optimum Heuristik
            Es werden Folgen gewertet.
            Die des Spielers positiv, die des Gegners negativ
        """  
        score = 0
        cases = a_dictionary
        keys = list(cases.keys())

        if spieler == 1:
            gegner = 2
        else: # spieler == 2
            gegner = 1

        for key in keys:
            anzahl = cases[key]
            stein = int(key[0])
            folge = key[2]
            if stein == spieler:
                score += self.score('spieler_'+folge) * anzahl
            elif stein == gegner:
                score += self.score('gegner_'+folge) * anzahl

        return score

    def evaluate_middle(self, spieler):
        """ Wertet gemäß der Optimum Heuristik die mittlere Spalte aus.
            Punkte liefert jedes eigene Element in der mittleren Spalte.
            Impliziert die Strategie, wenn möglich, die Mitte zu belegen.
            Liefert die Summe als int.
        """
        total = 0
        for piece in self.brett[:,3]:
            if piece == spieler:
                total += self.score('spieler_mid')
        return total   


    # Matthias
    def minmax(self, tiefe, spieler, heuristik_name, baum_printen = False, deterministic = True, pruning = True):
        """ Steuert die Auswahl der Heuristik und wählt den besten Zug 
            returns besten Zug
        """
        if heuristik_name == "random":
            heuristik_func = self.heuristik_zufall
        elif heuristik_name == "optimum":
            heuristik_func = self.heuristic_evaluate_optimum
        elif heuristik_name == "X":
            heuristik_func = self.heuristik_X
        else:
            heuristik_func = self.heuristic_evaluate_optimum
        if pruning: # Standard Suche mit pruning
            fits = self.get_fits(0, tiefe, spieler, heuristik_func, baum_printen=baum_printen)
        else:       # ohne Pruning
            self.baum = ["" for _ in range(tiefe+1)]
            fits = self.get_fits_without_pruning(0, tiefe, spieler, heuristik_func, baum_printen=baum_printen)
            fits.sort(key=lambda x: x[1], reverse=True) #den besten Zug nach vorne
            if baum_printen:
                print(str(fits[0][1]) + "\n" )
                print("\n\n".join(self.baum))
        
        if len(fits) == 0:
            return 999
        if deterministic:
            gewählter_zug = fits[0][0]
        else:
            weights = [x[1] for x in fits]
            minweight = min(weights)
            if minweight < 0:
                weights = [weight - minweight + 0.1 for weight in weights]
            sum_of_weights = sum(weights)
            if sum_of_weights == 0:
                normalized_weights = [1/len(fits) for weight in weights]
                sum_of_weights = 1
            else:
                normalized_weights = [weight/sum_of_weights for weight in weights]
            zug = random.choices(fits, weights = normalized_weights)
            gewählter_zug = zug[0][0]
        return gewählter_zug
   

    def get_fits_without_pruning(self,tiefe, S, spieler, heuristik_func, baum_printen= False):
        """ Steigt eine im Parameter definierte Tiefe ab, 
            ermittelt die Scores der jeweiligen Spielbretter.

            >ACHTUNG!!!! Für den Vergleich pruning <-> ohne pruning bei dieser Funktion hier eine Schranke eingeben, die um 1 geringer ist als bei pruning!!
        """
        fits = []
        for z in range(7):
            if self.zug_erlaubt(z):
                self.zug(z)
                if tiefe == S:
                    fits.append((z,heuristik_func(spieler, tiefe)))
                    self.undo()
                else:
                    fits_z = self.get_fits_without_pruning(tiefe+1, S, spieler, heuristik_func, baum_printen=baum_printen)
                    if tiefe %2 == 1: #MAX
                        max = fit_max(fits_z) # calls fit_max for MAXIMUM
                        fits.append((z, max))
                    else:             #MIN
                        min = fit_max(fits_z, reverse=False)
                        fits.append((z, min))
                    self.undo()
        if baum_printen:
            self.baum[tiefe] += "  ".join([f"{fit[1]}" for fit in fits]) + "    |    "
        return fits
    


    # waldemar
    def get_fits(self,tiefe, schranke, spieler, heuristik_func, baum_printen= False, a_alpha = - np.inf , a_beta = np.inf ):
        """ Steigt eine im Parameter definierte Tiefe ab, 
            ermittelt die Scores der jeweiligen Spielbretter.
            fits ist ein Array aus Tupeln (z=einwurfslot, heuristikbewertung).
            Wendet alpha, beta Pruning für eine bessere Performance an.
            Ausgehend von Tiefe 0 wird alphabeta_min zuerst aufgerufen.
            schranke = Schranke der Tiefe; tiefe = aktuelle Tiefe
        """
        fits = []           
        alpha = a_alpha    #initial neg. infinity 
        beta = a_beta      #initial infinity 
        if baum_printen:
            self.baum = ["" for _ in range(schranke+1)] 
            
        # mögliche Züge aufspannen und in der Tiefe mit alphabeta pruning untersuchen
        for z in range(7):
            if self.zug_erlaubt(z):
                self.zug(z)
                value = self.alphabeta_min(tiefe+1, schranke, spieler, heuristik_func, baum_printen, alpha, beta )
                self.undo()
                fits.append((z, value))

        if baum_printen:
            if tiefe %2 == 1:
                self.baum[tiefe] += "MAX "
            elif tiefe %2 == 0:
                self.baum[tiefe] += "MIN "
            self.baum[tiefe] += "  ".join([f"{fit[1]}" for fit in fits]) + "  " + " # A:" + str(alpha) + " B:" + str(beta) +  " |    "

        #fits absteigend sortieren
        fits.sort(key=lambda x: x[1], reverse=True) 

        if baum_printen:
            print(str(fits[0][1]) + "\n" )
            print("\n\n".join(self.baum))


        return fits  # gibt sortierte fits zurück, bestes fit an position 0

    def alphabeta_max(self, tiefe, schranke, spieler, heuristik_func, baum_printen= False, a_alpha = - np.inf , a_beta = np.inf ):
        """ Max Funktion des alpha beta pruning
        """
        alpha = a_alpha    #initial neg. infinity //    saves best max value
        beta = a_beta      #initial infinity //         saves best min value
        if tiefe >= schranke: 
            return heuristik_func(spieler, tiefe)

        if baum_printen:
            werte_it = []
                
        for z in range(7):
            if self.zug_erlaubt(z):
                self.zug(z)
                wert = self.alphabeta_min(tiefe+1, schranke, spieler, heuristik_func, baum_printen, alpha, beta )
                alpha = max(alpha, wert)
                self.undo()
                if baum_printen:
                    werte_it.append(wert)

                if beta < alpha:
                    return alpha

        if baum_printen:
            self.baum[tiefe] += "MIN "
            self.baum[tiefe] += "  ".join([f"{wert}" for wert in werte_it]) + "  " + " # A:" + str(alpha) + " B:" + str(beta) +  " |    "

        return alpha


    def alphabeta_min(self, tiefe, schranke, spieler, heuristik_func, baum_printen= False, a_alpha = - np.inf , a_beta = np.inf ):
        """ Min Funktion des alpha beta pruning
        """
        alpha = a_alpha    #initial neg. infinity //    saves best max value
        beta = a_beta      #initial infinity //         saves best min value
        if tiefe >= schranke: 
            return heuristik_func(spieler, tiefe)

        if baum_printen:
            werte_it = []
        
        for z in range(7):
            if self.zug_erlaubt(z):
                self.zug(z)
                wert = self.alphabeta_max(tiefe+1, schranke, spieler, heuristik_func, baum_printen, alpha, beta )
                beta = min(beta, wert)
                self.undo()
                if baum_printen:
                    werte_it.append(wert)
                    
                if beta < alpha:
                    return beta

        if baum_printen:
            self.baum[tiefe] += "MAX "
            self.baum[tiefe] += "  ".join([f"{wert}" for wert in werte_it]) + "  " + " # A:" + str(alpha) + " B:" + str(beta) +  " |    "

        return beta   

    # Hier speedbenchmark vorstellen

    def heuristik_X(self, spieler, schwierigkeit):
        spiel = VGW()
        spiel.brett=self.brett
        züge = self.letzter_zug
        spiel.letzter_zug=züge
        wichtige_züge = züge[-(schwierigkeit+1):]
        for wichtiger_zug in wichtige_züge:
            spiel.undo()
        total = 0
        total += self.evaluate_middle(spieler)

        multiplier = 100
        for wichtiger_zug in wichtige_züge:
            spiel.zug(wichtiger_zug[1],wichtiger_zug[2])
            all_lines = lines(spiel.brett)
            a_dictionary = eval_lines(all_lines)
            score = self.score_cases(a_dictionary, spieler)
            score *= multiplier
            total += score
            multiplier /=10
        return total

    def simulate_game(self, schwierigkeit1, heuristik1, schwierigkeit2, heuristik2, printgame = False, deterministic = False, pruning = False):
        """ Simuliert ein Spiel zwichen zwei KIs die beide eine Schwierigkeit (Tiefe des Baums) und eine Heuristik zugewiesen kriegen.
        Wenn printgame True ist wird der Spielverlauf während der Simulation geprinted.
        Wenn deterministic auf False steht arbeitet der Algorithmus nicht deterministisch in der Form, dass ein Weighted Random Choice mithilfe der Scores gemacht wird. 
        """
        spieler1 = 1
        spieler2 = 2
        while True:
            zug1 = self.simulate_zug(schwierigkeit1, spieler1, heuristik1, printgame, deterministic, pruning=pruning)
            if zug1:
                return zug1
            zug2 = self.simulate_zug(schwierigkeit2, spieler2, heuristik2, printgame, deterministic, pruning=pruning)
            if zug2:
                return zug2


    def simulate_zug(self, schwierigkeit, spieler, heuristik, printgame = False, deterministic = False, pruning = False):
        spieler_zug = self.minmax(schwierigkeit, spieler, heuristik, deterministic=deterministic, pruning=pruning)
        if spieler_zug == 999:
            print("Unentschieden!!")
            return 999
        self.zug(spieler_zug, spieler)
        if printgame:
            sleep(0.5)
            os.system("clear")
            print(self)
        color = colored("x", "red") if spieler == 1 else colored("o", "blue")
        if self.gewinner_ermitteln():
            print(f"SPIEL VORBEI. Spieler {spieler}("+color+")hat gewonnen!!!")
            print(self)
            return spieler
        return False

def fit_max(fits, reverse = True):
    fits.sort(key=lambda x: x[1], reverse=reverse)
    if len(fits)==0:
        return 999
    return fits[0][1]
def diags(a):
    """ Gibt alle mögliche Diagonalen aus dem Spielfeld zurück die mindestens die Länge 4 haben """
    # Credits to https://stackoverflow.com/a/6313414
    diags = [a[::-1,:].diagonal(i) for i in range(-a.shape[0]+1,a.shape[1])]
    diags.extend(a.diagonal(i) for i in range(a.shape[1]-1,-a.shape[0],-1))
    return [n for n in diags if len(n) >=4]

def lines(a):
    """ Gibt ein Array aus allen möglichen horizontalen, vertikalen und diagonalen Reihen des Spielbretts zurück."""
    return [x for x in a]+[x for x in a.T]+diags(a)

def eval_lines(lines):
    """ Zählt im kompletten Spielfeld die Anzahl der zusammenhängenden Folgen pro Spieler."""
    counts = defaultdict(int)
    for line in lines:
        counts_line = eval_line(line)
        for cl in counts_line:
            counts[cl] += counts_line[cl]
    return counts

def eval_line(line):
    """ Zählt in einer gegebenen Reihe die Anzahl der zusammenhängenden Folgen pro Spieler."""
    counts = defaultdict(int)
    last = 0
    row = 0
    for value in line:
        if value != last or value == 0:
            row = 0
        elif value > 0:
            row += 1
            for r in range(1, row+1):
                length = r+1 if r+1 <= 4 else 4
                counts[f"{int(last)}_{length}"]+=1
        last = value
    return counts


if __name__ == "__main__":
    spiel = VGW()
    # Simuliere 10 zufällige Spielzüge.
    for x in range(16):
        spiel.zug(random.randint(0,6))

    # Um das Spiel 'Schön' auszugeben einfach print(spiel)
    print(spiel)
    # Um die Spiel Matrix auszugeben print(spiel.brett)
    print(spiel.brett)
    # Um Eine Situation auf dem Brett auszuwerten:
    print(eval_lines(lines(spiel.brett)))
    # Gibt ein Dictionary zurück mit Einträgen 'spieler_reihe:anzahl'
    # 2_4:3 würde also bedeuten dass Spieler2 3 Reihen hat mit Länge 4.

    """
    spiel.set_to_state(
        (([[0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 2., 1., 1., 1., 0., 1.],
       [0., 1., 2., 2., 1., 0., 2.],
       [0., 2., 1., 2., 2., 2., 1.]]), [(0, 0, 0), (5, 1, 2), (4, 1, 1), (5, 5, 2), (5, 2, 1), (5, 4, 2), (5, 6, 1), (4, 2, 2), (4, 4, 1), (4, 6, 2), (3, 6, 1), (3, 4, 1), (5, 3, 2), (3, 2, 1), (4, 3, 2), (3, 3, 1), (3, 1, 2), (2, 3, 1)])
    )
    """

    print("Spielbrett Bewertung: " + str(spiel.heuristic_evaluate_optimum(1)) )

    """
    if not spiel.gewinner_ermitteln():
        print("noch kein gewinner")
        spiel.single_player(1, "optimum",baum_printen=True)
    """
    #AUSWAHL = spiel.minmax(4,1,"optimum",baum_printen=False,deterministic=True,pruning=True )
    

    print("END OF TEST")