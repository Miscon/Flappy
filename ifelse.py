from ple.games.flappybird import FlappyBird
from ple import PLE
import random

class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        return
    
    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation
        return

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1) 

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        # print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        pipe_center_y = state["next_pipe_top_y"] + 65
        next_pipe_center_y = state["next_next_pipe_top_y"] + 65
        player_y = state["player_y"]
        distance_to_pipe = state["next_pipe_dist_to_player"]
        distance_to_next_pipe = state["next_next_pipe_dist_to_player"]
        player_velocity = state["player_vel"]
        
        action = 0
        # if distance_to_pipe > 79: # if still inside previous pipe, keep velocity stable
        #     if player_velocity > 0: # if velocity is positive, then flappy is moving down
        #         action = 0
        #     else: # flappy is moving up
        #         action = 1
        
        difference = player_y - pipe_center_y
        if distance_to_pipe < 10:
            # print(distance_to_next_pipe)
            difference = player_y - next_pipe_center_y
        if difference < 0:
            action = 1
        # if player_velocity < 0 and  # check if about to crash into current pipe
        # print("difference: {},   velocity: {},    action: {}".format(difference, player_velocity, action))


        return action



def play():
    agent = FlappyAgent()    

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}

    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None, reward_values = reward_values)
    env.init()

    score = 0
    frames = 0
    nb_episodes = 10
    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        action = agent.policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)
        frames += 1
        score += reward
        if reward == 1:
            # print("----------- pipe passed ----------")
            pipe_center_y = state["next_pipe_top_y"] + 65
            next_pipe_center_y = state["next_next_pipe_top_y"] + 65
            player_y = state["player_y"]
            distance_to_pipe = state["next_pipe_dist_to_player"]
            distance_to_next_pipe = state["next_next_pipe_dist_to_player"]
            player_velocity = state["player_vel"]
            difference = player_y - pipe_center_y
            # print("difference: {}".format(difference))
            # print("pipe_center_y: {}".format(pipe_center_y))
            # print("next_pipe_center_y: {}".format(next_pipe_center_y))
            # print("distance_to_pipe: {}".format(distance_to_pipe))
            # print("distance_to_next_pipe: {}".format(distance_to_next_pipe))
            # print(frames)

            import time
            # time.sleep(2)

        if state["player_y"] > state["next_pipe_top_y"] - 6 and state["player_y"] < state["next_pipe_top_y"] + 6:

            import time
            # print(state["player_y"])
            # print(state["next_pipe_top_y"])
            # time.sleep(3)

        if state["next_pipe_dist_to_player"] < 75 and state["next_pipe_dist_to_player"] > 70:

            import time
            time.sleep(3)

        # reset the environment if the game is over
        if env.game_over():
            print("Score: {}".format(score))
            env.reset_game()
            nb_episodes -= 1
            score = 0

    print(frames)


play()