import copy
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


# Rotates postions and actions of players within given state, returns flipped state
# Notes
#   Rotating about y axis changes direction of play for this state.
#   Rotating about x axis doesn't change direction of play, but helps to explore state space.
#   Flipping state can produce the same state, if all positon/velocity/actions
#       aren't on a flipped axis.
def flip_state(state, flip_x=True, flip_y=True):
    temp = copy.deepcopy(state)
    temp = flip_state_positions(temp, flip_x=flip_x, flip_y=flip_y)
    temp = flip_state_actions(temp, flip_x=flip_x, flip_y=flip_y)
    return temp

def flip_state_positions(state, flip_x=True, flip_y=True):
    # flips the axis if the correspong axis flip is True, otherwise stays the 
    x_mult = -1 if flip_x else 1
    y_mult = -1 if flip_y else 1
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

def flip_state_actions(state, flip_x=True, flip_y=True):
    for i in range(len(state.players)):
        inp = state.players[i].input
        if inp == Input.Up and flip_y:
            state.players[i].input = Input.Down
        elif inp == Input.Down and flip_y:
            state.players[i].input = Input.Up

        if inp == Input.Left and flip_x:
            state.players[i].input = Input.Right
        elif inp == Input.Right and flip_x:
            state.players[i].input = Input.Left

    return state












