import copy
import random
from replay import Replay, State, Input

def filter_states(game_states):
    processed_states = []

    for state, player_team in game_states:
        our_player = state.players[0] if state.players[0].team == player_team else state.players[1]

        # Remove states where game is not being played e.g. Menu, Pause, Goal etc.
        if state.state != State.Game:
            continue

        # Get all states where the ball is kicked
        if our_player.input[4]:
            processed_states.append((state, player_team))
            continue

        # Randomization based filtering
        if sum(our_player.input) == 0 and random.random() < 0.8:
            processed_states.append((state, player_team))

        if sum(our_player.input) == 1 and random.random() < 0.5:
            processed_states.append((state, player_team))

        if sum(our_player.input) == 2 and random.random() < 0.5:
            processed_states.append((state, player_team))

        # Remove inconsistent game states
        if sum(our_player.input) == 3:
            if our_player.input[0] and our_player.input[1]:
                continue

            if our_player.input[2] and our_player.input[3]:
                continue

            processed_states.append((state, player_team))

    return processed_states


def filter_states_3v3DC(game_states):
    processed_states = []
    for state_dict in game_states:
        if True in state_dict['outputs']:
            processed_states.append(state_dict)

    return processed_states



def get_players(state, player_team):
    our_team = []
    opp_team = []
    for i in range(len(state.players)):
        if state.players[i].team == player_team:
            our_team.append(state.players[i])
        else:
            opp_team.append(state.players[i])
    return (our_team, opp_team)


# Rotates postions and actions of players within given state, returns flipped state
# Notes
#   Rotating about y axis changes direction of play for this state.
#   Rotating about x axis doesn't change direction of play, but helps to explore state space.
#   Flipping state can produce the same state, if all positon/velocity/actions
#       aren't on a flipped axis.
def flip_state(state, x_axis_flip=True, y_axis_flip=True):
    temp = copy.deepcopy(state)
    temp = flip_state_positions(temp, x_axis_flip=x_axis_flip, y_axis_flip=y_axis_flip)
    temp = flip_state_actions(temp, x_axis_flip=x_axis_flip, y_axis_flip=y_axis_flip)
    return temp

def flip_state_positions(state, x_axis_flip=True, y_axis_flip=True):
    # changes variable is other variable axis flip is True, otherwise stays the same
    x_mult = -1 if y_axis_flip else 1
    y_mult = -1 if x_axis_flip else 1
    for i in range(len(state.players)):
        state.players[i].disc.x *= x_mult
        state.players[i].disc.y *= y_mult
        state.players[i].disc.vx *= x_mult
        state.players[i].disc.vy *= y_mult

    state.ball.x *= x_mult
    state.ball.y *= y_mult
    state.ball.vx *= x_mult
    state.ball.vy *= y_mult

    return state

def flip_state_actions(state, x_axis_flip=True, y_axis_flip=True):
    for i in range(len(state.players)):
        inp = state.players[i].input
        if inp == Input.Up and x_axis_flip:
            state.players[i].input = Input.Down
        elif inp == Input.Down and x_axis_flip:
            state.players[i].input = Input.Up

        if inp == Input.Left and y_axis_flip:
            state.players[i].input = Input.Right
        elif inp == Input.Right and y_axis_flip:
            state.players[i].input = Input.Left

    return state


def flip_action_list(al, x_axis_flip=True, y_axis_flip=True):
    for idx in range(len(al)):
        inp = al[idx]
        if inp == Input.Up and x_axis_flip:
            al[idx] = Input.Down
        elif inp == Input.Down and x_axis_flip:
            al[idx] = Input.Up

        if inp == Input.Left and y_axis_flip:
            al[idx] = Input.Right
        elif inp == Input.Right and y_axis_flip:
            al[idx] = Input.Left
    return al












