from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy as np

class FlappyAgent:
    def __init__(self):
        self.Q = {} # see picture or rl2, averaging learning rule
        self.greedy_policy = {}
        self.episode = []
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
        # when we learn with on policy monte carlo we need the whole episode and then we can update
        # TODO: learn from the observation
        # 1. remember these statetransitions = episode.append((s1, a, r))

        # TODO turn state into set first
        s1 = (s1["player_y"], s1["player_vel"], s1["next_pipe_dist_to_player"], s1["next_pipe_bottom_y"])
        self.episode.append((s1, a, r))

        # 2. if reached the end of episode, learn from it, then reset = episode = [] part(b)
        if end:
            self.learn_from_episode(self.episode)
            self.episode = []

        return

    def learn_from_episode(self, episode):        
        #instead of append G to returns, use the update rule # check photo on phone"
        a = 0.1 # TODO learning rate, set correctly
        gamma = 0.9 # TODO discount, set correctly
        states = set()
        
        # b)
        G = 0
        for s, a, r in reversed(episode):
            print(type(s))
            G = r + gamma * G
            self.Q[(s, a)] = self.Q[(s, a)] + a * (G - self.Q[(s, a)]) # update rule
            states.add(s)
        
        # c)
        for s in states: 
            A_star = None
            ret = None
            # A more generalized loop over actions even though they are just two
            for action in [0, 1]:
                q_value = self.Q.get((s, action)) # get returns None if none exists
                if q_value is not None:
                    if ret is None:
                        A_star = action
                        ret = q_value
                    else:
                        if ret < q_value:
                            A_star = action
                            ret = q_value
            if A_star is not None:
                self.greedy_policy[s] = A_star


        

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """


        print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.

        # with epsilon chance, pick a random action
        # else the policy greedy action
        # only initialize states when you see them, to 0

        e = 0.1
        greedy = np.random.choice([False, True], p=[e, 1-e])

        action = self.greedy_policy.get(state) # get returns None if none exists
        if greedy or action is None:
            action = random.randint(0, 1)

        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        action = self.greedy_policy.get(state) # get returns None if none exists
        if action is None:
            action = random.randint(0, 1)

        return action

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    training = True

    if training:
        reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
        # reward_values = agent.reward_values
        
        env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
                reward_values = reward_values)
        # TODO: to speed up training change parameters of PLE as follows:
        # display_screen=False, force_fps=True 
        env.init()

        score = 0
        while nb_episodes > 0:
            # pick an action
            state1 = env.game.getGameState()
            action = agent.training_policy(state1)

            # step the environment
            reward = env.act(env.getActionSet()[action])
            print("reward=%d" % reward)

            state2 = env.game.getGameState()

            agent.observe(state1, action, reward, state2, env.game_over())

            score += reward
        
        else:
            reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
            
            env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
                    reward_values = reward_values)
            env.init()

            score = 0
            while nb_episodes > 0:
                # pick an action
                action = agent.policy(state1)

                # step the environment
                reward = env.act(env.getActionSet()[action])
                print("reward=%d" % reward)

                score += reward


        # reset the environment if the game is over
        if env.game_over():
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0

agent = FlappyAgent()
run_game(1, agent)
