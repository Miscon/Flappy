from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import numpy as np
import pickle
import scipy.stats as st
import pandas as pd



class FlappyAgent:
    def __init__(self):
        self.Q = {} # see picture or rl2, averaging learning rule
        self.greedy_policy = {}
        self.s_a_counts = {} # for graphs
        self.episode = []
  
        self.episode_count = 0 # for graphs
        self.frame_count = 0 # for graphs
        return
    
    def map_state(self, state):
        """ We are not using the entire game state as our state as we 
            just want the following values but discretized to at max 15 values
            player_y
            next_pipe_top_y
            next_pipe_dist_to_player
            player_vel
        """

        player_y = int(state['player_y'] / (513.0/15))
        player_vel = int(state['player_vel'] / (20.0/15))
        next_pipe_dist_to_player = int(state['next_pipe_dist_to_player'] / (289.0/15))
        next_pipe_top = int(state['next_pipe_top_y'] / (513.0/15))

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
        
        lr = 0.1 # learning rate
        gamma = 1 # discount
        states = set()
        
        # b)
        # instead of append G to returns, use the update rule
        G = 0
        for s, a, r in reversed(episode):
            G = r + gamma * G
            if (s, a) in self.Q:
                self.Q[(s, a)] = self.Q[(s, a)] + lr * (G - self.Q[(s, a)]) # update rule
            else:
                self.Q[(s, a)] = G
            states.add(s)
        
        # c)
        for s in states: 

            action = None
            G0 = self.Q.get((s, 0))
            G1 = self.Q.get((s, 1))

            if G0 == None and G1 == None:
                pass
            elif G0 == None:
                action = 0
            elif G1 == None:
                action = 1
            else:
                if G0 > G1:
                    action = 0
                else:
                    action = 1

            if action is not None:
                self.greedy_policy[s] = action

        

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

        action = self.greedy_policy.get(state) # get returns None if none exists
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
        action = self.greedy_policy.get(state) # get returns None if none exists
        if action is None:
            action = random.randint(0, 1)

        return action


def train_agent():
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    agent = FlappyAgent()

    try:
        with open("on_policy_monte_carlo/newest.pkl", "rb") as f:
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
    count = 0
    rewards = []
    nb_episodes = 10000
    while nb_episodes > 0 and agent.frame_count <= 2000000:
        # pick an action
        state1 = env.game.getGameState()
        action = agent.training_policy(state1)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        # print("reward=%d" % reward)

        state2 = env.game.getGameState()

        agent.observe(state1, action, reward, state2, env.game_over())

        score += reward
        count += 1

        # reset the environment if the game is over
        if env.game_over():
            env.reset_game()
            nb_episodes -= 1

            rewards.append(score)
            if nb_episodes % 100 == 0:
                print("==========================")
                print("episodes done: {}".format(agent.episode_count))
                print("episodes left: {}".format(nb_episodes))
                print("frames: {}".format(count))

                with open("on_policy_monte_carlo/{}.pkl".format(agent.episode_count), "wb") as f:
                    pickle.dump((agent), f, pickle.HIGHEST_PROTOCOL)
                with open("on_policy_monte_carlo/newest.pkl", "wb") as f:
                    pickle.dump((agent), f, pickle.HIGHEST_PROTOCOL)
            score = 0
         

def score_agent():

    agent = FlappyAgent()
    episode_snapshot = 100
    avg_scores = []
    df = pd.DataFrame()
    while True:
        try:
            with open("on_policy_monte_carlo/{}.pkl".format(episode_snapshot), "rb") as f:
                agent = pickle.load(f)
                print("Running snapshot {}".format(agent.episode_count))
        except:
            break

        reward_values = agent.reward_values()

        env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None, reward_values = reward_values)
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
            if env.game_over():
                scores.append(score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
        
        avg_score = sum(scores) / float(len(scores))
        confidence_interval = st.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))  
        df = df.append({"episode_count":agent.episode_count, "frame_count":agent.frame_count,
                        "avg_score":avg_score, "interval_lower":confidence_interval[0],
                        "interval_upper":confidence_interval[1]}, ignore_index=True)
        print(avg_score)
        episode_snapshot += 100

    df.to_csv("on_policy_monte_carlo/scores.csv", encoding='utf-8', index=False)


def play(nb_episodes):
    agent = FlappyAgent()
    
    with open("on_policy_monte_carlo/newest.pkl", "rb") as f:
        agent = pickle.load(f)
        print("Running snapshot {}".format(agent.episode_count))
    

    reward_values = agent.reward_values()

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

score_agent()

play(20)