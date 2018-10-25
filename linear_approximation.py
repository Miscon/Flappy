from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy as np
import pickle
import scipy.stats as st
import pandas as pd



class FlappyAgent:
    def __init__(self):
        self.Q = {}
        self.episode = []
        self.weights = [np.array([0]*4, dtype=np.float64), np.array([0]*4, dtype=np.float64)] # [0 flap, 0 noop]
        
  
        #for graphs
        self.s_a_counts = {}
        self.episode_count = 0
        self.frame_count = 0
    
    def map_state(self, state):
        """ 
            player_y
            next_pipe_top_y
            next_pipe_dist_to_player
            player_vel
        """

        player_y = state['player_y']
        player_vel = state['player_vel']
        next_pipe_dist_to_player = state['next_pipe_dist_to_player']
        next_pipe_top = state['next_pipe_top_y']

        return (player_y, player_vel, next_pipe_dist_to_player, next_pipe_top)

    
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
        # 1. remember these statetransitions = episode.append((s1, a, r))

        s1 = self.map_state(s1)
        self.episode.append((s1, a, r))
        
        # count for graphs
        if (s1, a) in self.s_a_counts:
            self.s_a_counts[(s1, a)] += 1
        else:
            self.s_a_counts[(s1, a)] = 1
        
        self.frame_count += 1

        # 2. if reached the end of episode, learn from it, then reset = episode = [] part(b)
        if end:
            self.learn_from_episode(self.episode)
            self.episode_count += 1
            self.episode = []

        return

    def learn_from_episode(self, episode):        
        
        lr = 0.00001 # learning rate
        gamma = 1 # discount
        
        # b)
        # instead of append G to returns, use the update rule
        G = 0
        for s, a, r in reversed(episode):
            G = r + gamma * G
            features = np.array(s, dtype=np.float64)
            self.weights[a] = self.weights[a] - lr * (np.dot(self.weights[a], features) - G) * features

                
        # print("--=-=-=-=-=-=-=-=-=-=-=-=--")
        # print("weights 0: ", self.weights[0])
        # print("weights 1: ", self.weights[1])
        # print("--=-=-=-=-=-=-=-=-=-=-=-=--")


        


        # G = 0
        # for s, a, r in reversed(episode):
        #     G = r + gamma * G
        #     if (s, a) in self.Q:
        #         self.Q[(s, a)] = self.Q[(s, a)] + lr * (G - self.Q[(s, a)]) # update rule
        #     else:
        #         self.Q[(s, a)] = G # initialize to the first example
    

    def get_state_max_a(self, s):

        features = np.array(s, dtype=np.float64)

        G0 = np.dot(self.weights[0], features)
        G1 = np.dot(self.weights[1], features)

        if G0 == None and G1 == None:
            return None
        elif G0 == None:
            return G0
        elif G1 == None:
            return G1
        else:
            if G0 > G1:
                return G0
            else:
                return G1


    def get_state_argmax_a(self, s):

        features = np.array(s, dtype=np.float64)

        G0 = np.dot(self.weights[0], features)
        G1 = np.dot(self.weights[1], features)

        if G0 == None and G1 == None:
            return None
        elif G0 == None:
            return 0
        elif G1 == None:
            return 1
        else:
            if G0 > G1:
                return 0
            else:
                return 1        

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """


        # print("state: %s" % state)
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.

        # with epsilon chance, pick a random action
        # else the policy greedy action
        # only initialize states when you see them, to 0
        state = self.map_state(state)

        e = 0.1 # epsilon - exploration
        greedy = np.random.choice([False, True], p=[e, 1-e])

        action = self.get_state_argmax_a(state) # get returns None if none exists
        if not greedy or action is None:
            action = random.randint(0, 1)

        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        # print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        state = self.map_state(state)
        action = self.get_state_argmax_a(state) # get returns None if none exists
        if action is None:
            action = random.randint(0, 1)

        return action


def train_agent():
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    agent = FlappyAgent()

    try:
        with open("linear_approximation/newest.pkl", "rb") as f:
            agent = pickle.load(f)
            print("Agent starting from episode {}".format(agent.episode_count))
    except:
        print("Agent not found, starting new agent")
    # reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True , rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    rewards = []
    nb_episodes = 10000 - agent.episode_count
    while nb_episodes > 0 and agent.frame_count <= 2000000:
        # pick an action
        state1 = env.game.getGameState()
        action = agent.training_policy(state1)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        state2 = env.game.getGameState()

        end = env.game_over() or score >= 100 # Stop after reaching 100 pipes
        agent.observe(state1, action, reward, state2, end)

        score += reward

        # reset the environment if the game is over
        if end:
            env.reset_game()
            nb_episodes -= 1

            rewards.append(score)
            if nb_episodes % 100 == 0:
                print("episodes done: {}".format(agent.episode_count))
                print("episodes left: {}".format(nb_episodes))
                print("frames: {}".format(agent.frame_count))

                score_agent(agent)

                with open("linear_approximation/{}.pkl".format(agent.episode_count), "wb") as f:
                    pickle.dump((agent), f, pickle.HIGHEST_PROTOCOL)
                with open("linear_approximation/newest.pkl", "wb") as f:
                    pickle.dump((agent), f, pickle.HIGHEST_PROTOCOL)
                
                print("==========================")

            score = 0
         

def score_agent(agent):

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}

    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None, reward_values=reward_values)
    env.init()

    score = 0
    scores = []
    nb_episodes = 50
    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        action = agent.policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        score += reward

        # reset the environment if the game is over
        if env.game_over() or score >= 100:
            scores.append(score)
            env.reset_game()
            nb_episodes -= 1
            score = 0
    
    avg_score = sum(scores) / float(len(scores))
    confidence_interval = st.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))  
    
    df = pd.DataFrame()
    try:
        df = pd.read_csv("linear_approximation/scores.csv")
    except:
        print("Starting new scoring file")

    df = df.append({"episode_count":agent.episode_count, "frame_count":agent.frame_count,
                    "avg_score":avg_score, "interval_lower":confidence_interval[0],
                    "interval_upper":confidence_interval[1]}, ignore_index=True)
    df.to_csv("linear_approximation/scores.csv", encoding='utf-8', index=False)


def play():
    agent = FlappyAgent()
    
    with open("linear_approximation/newest.pkl", "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
    

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}

    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None, reward_values = reward_values)
    env.init()

    score = 0

    nb_episodes = 50
    while nb_episodes > 0:
        # pick an action
        state = env.game.getGameState()
        action = agent.policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        score += reward

        # reset the environment if the game is over
        if env.game_over():
            print("Score: {}".format(score))
            env.reset_game()
            nb_episodes -= 1
            score = 0


# train_agent()
play()