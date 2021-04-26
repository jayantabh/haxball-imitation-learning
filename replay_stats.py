import os
from replay import Replay, State

def main():
    xminmax = [0,0]
    yminmax = [0,0]
    bin_file_path = 'preprocessed'
    
    for root, dirs, files in os.walk(bin_file_path):
        for name in files:
            if name.split('.')[-1] == "bin":
                print(name)
                with open(os.path.join(bin_file_path, name), 'rb') as f:
                    file_content = f.read()
                    _, states = Replay(file_content)

                    for state in states:
                        if state.players is not None and len(state.players) == 2:
                            xmin = state.players[0].disc.x if state.players[0].disc.x < state.players[1].disc.x else state.players[1].disc.x
                            xmax = state.players[0].disc.x if state.players[0].disc.x > state.players[1].disc.x else state.players[1].disc.x

                            ymin = state.players[0].disc.y if state.players[0].disc.y < state.players[1].disc.y else state.players[1].disc.y
                            ymax = state.players[0].disc.y if state.players[0].disc.y > state.players[1].disc.y else state.players[1].disc.y

                            xminmax[0] = xmin if xmin < xminmax[0] else xminmax[0]
                            xminmax[1] = xmax if xmax > xminmax[1] else xminmax[1]

                            yminmax[0] = ymin if ymin < yminmax[0] else yminmax[0]
                            yminmax[1] = ymax if ymax > yminmax[1] else yminmax[1]
                    print('x min max:', xminmax)
                    print('y min max:', yminmax)
                print('---------------------')


main()


