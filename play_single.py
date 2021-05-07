from Spiel import VGW
spiel = VGW()
spiel.single_player(3, "optimum", baum_printen=False, deterministic=False, pruning=False)
