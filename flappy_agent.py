from ple.games.flappybird import FlappyBird
from ple import PLE
import random

import pickle
import pandas as pd
import numpy as np
import scipy.stats as st
import os.path
import os
import matplotlib.pyplot as plt
import seaborn as sns
import threading



class FlappyAgent:
    def __init__(self, name):
        self.name = name
        self.Q = {}
        self.initial_return_value = 0

        self.lr = 0.1 # learning rate
        self.e = 0.1 # epsilon / exploration
        self.gamma = 1 # discount
 
        # For graphs
        self.s_a_counts = {}
        self.episode_count = 0
        self.frame_count = 0
    

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    

    def map_state(self, state):
        return state


    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        pass


    def get_max_a(self, state):
        G0 = self.Q.get((state, 0))
        G1 = self.Q.get((state, 1))

        if G0 is None:
            G0 = self.initial_return_value
        if G1 is None:
            G1 = self.initial_return_value

        if G0 > G1:
            return G0
        else:
            return G1


    def get_argmax_a(self, state):
        G0 = self.Q.get((state, 0))
        G1 = self.Q.get((state, 1))

        if G0 is None:
            G0 = self.initial_return_value
        if G1 is None:
            G1 = self.initial_return_value

        if G0 == G1:
            return random.randint(0, 1)
        elif G0 > G1:
            return 0
        else:
            return 1


    def update_counts(self, sa):
        if sa in self.s_a_counts:
            self.s_a_counts[sa] += 1
        else:
            self.s_a_counts[sa] = 1
        self.frame_count += 1
       

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """

        state = self.map_state(state)

        greedy = np.random.choice([False, True], p=[self.e, 1-self.e])

        action = 0
        if greedy:
            action = self.get_argmax_a(state)
        else:
            action = random.randint(0, 1)

        
        return action


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """

        state = self.map_state(state)
        action = self.get_argmax_a(state)
        return action


    def draw_plots(self, once=False):

        f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        while True:
            try:
                d = {}
                i = 0
                for sa, G in self.Q.items():
                    
                    state = sa[0]
                    d[i] = {"player_y":state[0],
                            "player_vel":state[1],
                            "next_pipe_dist_to_player":state[2],
                            "next_pipe_top_y":state[3],
                            "y_difference":state[0] - state[3],
                            "action":self.get_argmax_a(state),
                            "return":G,
                            "count_seen":self.s_a_counts[sa]}
                    i += 1

                df = pd.DataFrame.from_dict(d, "index")
                df = df.pivot_table(index="y_difference", columns="next_pipe_dist_to_player", values=["action", "return", "count_seen"])
                
                ax1.cla()
                ax2.cla()
                ax3.cla()
                ax4.cla()
                self.plot_learning_curve(ax1)
                self.plot_actions(ax2, df)
                self.plot_expected_returns(ax3, df)
                self.plot_states_seen(ax4, df)
                
                if once:
                    plt.show()
                    break
                else:
                    plt.pause(5)
            except:
                pass
            

    def plot_learning_curve(self, ax):
        
        df = pd.read_csv("{}/scores.csv".format(self.name))

        ax.plot(df["frame_count"], df["avg_score"])
        ax.fill_between(df["frame_count"], df["interval_upper"], df["interval_lower"], alpha=0.5)

        ax.set_xlabel("Number of frames")
        ax.set_ylabel("Average score")
        ax.set_title("{}\nAverage score from 10 games".format(self.name.replace("_", " ")))


    def plot_actions(self, ax, df):
        ax.pcolor(df["action"])
        ax.set_title("Best action")
        
        # Invert axes
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])


    def plot_expected_returns(self, ax, df):
        ax.pcolor(df["return"])
        ax.set_title("Expected return")
        
        # Invert axes
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])


    def plot_states_seen(self, ax, df):
        ax.pcolor(df["count_seen"])
        ax.set_title("State count seen")
        
        # Invert axes
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_ylim(ax.get_ylim()[::-1])


    def run(self, arg):
        if arg == "train":
            self.train()
        elif arg == "play":
            self.play()
        elif arg == "score":
            self.score(False, 100)
        elif arg == "plot":
            self.draw_plots(True)
        else:
            print("Invalid argument, use 'train', 'play' or 'plot'")


    def train(self):
        """ Runs nb_episodes episodes of the game with agent picking the moves.
            An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
        """

        if not os.path.exists(self.name):
            os.mkdir(self.name)

        t = threading.Thread(target=self.draw_plots)
        t.daemon = True
        t.start()

        reward_values = self.reward_values()        
        env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True , rng=None, reward_values=reward_values)
        env.init()

        score = 0
        while self.frame_count <= 1000000:
            # pick an action
            state1 = env.game.getGameState()
            action = self.training_policy(state1)

            # step the environment
            reward = env.act(env.getActionSet()[action])
            # print("reward=%d" % reward)

            state2 = env.game.getGameState()

            end = env.game_over() or score >= 100 # Stop after reaching 100 pipes
            self.observe(state1, action, reward, state2, end)

            # reset the environment if the game is over
            if end:
                env.reset_game()
                score = 0

            if self.frame_count % 1000 == 0:
                print("==========================")
                
                print("episodes done: {}".format(self.episode_count))
                print("frames done: {}".format(self.frame_count))

                self.score(10)

                with open("{}/agent.pkl".format(self.name), "wb") as f:
                    pickle.dump((self), f, pickle.HIGHEST_PROTOCOL)

                print("==========================")


    # TODO delete this
    # def pos_check(self, env, state):
    #     import pygame
    #     from pygame.locals import *
    #     # self.died_pos(state1, state2)

    #     player_y = state["player_y"]
    #     player_vel = state["player_vel"]
    #     pipe_top_y = state["next_pipe_top_y"]
    #     next_pipe_top_y = state["next_next_pipe_top_y"]
    #     distance_to_pipe = state["next_pipe_dist_to_player"]
    #     next_distance_to_pipe = state["next_next_pipe_dist_to_player"]

    #     player_pipe_difference = player_y - pipe_top_y
    #     pipe_next_pipe_difference = pipe_top_y - next_pipe_top_y
        
    #     if distance_to_pipe > 30 and distance_to_pipe < 60:
    #         pipe_group = env.game.pipe_group
    #         hit = pygame.sprite.spritecollide(env.game.player, pipe_group, False)
    #         for h in hit:    #do check to see if its within the gap.
    #             top_pipe_check = ((env.game.player.pos_y - env.game.player.height/2) <= h.gap_start) 
    #             bot_pipe_check = ((env.game.player.pos_y + env.game.player.height) > h.gap_start+env.game.pipe_gap)

    #             # if top_pipe_check:
    #             #     print("top")
    #             #     print("player: {}".format(env.game.player.pos_y - env.game.player.height/2))
    #             #     print("player: {}".format(env.game.player.pos_y))
    #             #     print("player: {}".format(player_y))
    #             #     print("pipe: {}".format(h.gap_start))
    #             #     print("pipe: {}".format(h.gap_start - pipe_top_y))
    #             #     print("==========================")

    #             if bot_pipe_check:
    #                 print("bottom")
    #                 print("player: {}".format(env.game.player.pos_y + env.game.player.height))
    #                 print("player: {}".format(env.game.player.pos_y))
    #                 print("player: {}".format(player_y))
    #                 print("pipe: {}".format(h.gap_start+env.game.pipe_gap))
    #                 print("pipe: {}".format(h.gap_start+env.game.pipe_gap  - state["next_pipe_bottom_y"]))
    #                 print("==========================")

    # def pipe_check(self, env, state):
    #     import pygame
    #     from pygame.locals import *
    #     # self.died_pos(state1, state2)

    #     player_y = state["player_y"]
    #     player_vel = state["player_vel"]
    #     pipe_top_y = state["next_pipe_top_y"]
    #     next_pipe_top_y = state["next_next_pipe_top_y"]
    #     distance_to_pipe = state["next_pipe_dist_to_player"]
    #     next_distance_to_pipe = state["next_next_pipe_dist_to_player"]

    #     pipe_group = env.game.pipe_group
    #     # hit = pygame.sprite.spritecollide(env.game.player, pipe_group, False)
    #     # for h in hit:    #do check to see if its within the gap.
    #         # print(h.rect)

    #     # for pipe in pipe_group:
    #     #     print(pipe.rect)
    #     if player_y < pipe_top_y or player_y > state["next_pipe_bottom_y"]:
    #         pipe_group = env.game.pipe_group
    #         hit = pygame.sprite.spritecollide(env.game.player, pipe_group, False)
    #         for h in hit:    #do check to see if its within the gap.
    #             top_pipe_check = ((env.game.player.pos_y - env.game.player.height/2) <= h.gap_start) 
    #             bot_pipe_check = ((env.game.player.pos_y + env.game.player.height) > h.gap_start+env.game.pipe_gap)

    #             if top_pipe_check or bot_pipe_check:
    #                 print(distance_to_pipe)




    def score(self, training=True, nb_episodes=10):
        reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}

        env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None, reward_values=reward_values)
        env.init()

        total_episodes = nb_episodes
        score = 0
        scores = []
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = self.policy(state)

            # step the environment
            reward = env.act(env.getActionSet()[action])

            score += reward

            # reset the environment if the game is over
            if env.game_over() or score >= 50:
                scores.append(score)
                env.reset_game()
                nb_episodes -= 1
                score = 0
                # print(nb_episodes)
        
        avg_score = sum(scores) / float(len(scores))
        confidence_interval = st.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))  
        if np.isnan(confidence_interval[0]):
            confidence_interval = (avg_score, avg_score)
        
        print("Games played: {}".format(total_episodes))
        print("Average score: {}".format(avg_score))
        print("95 confidence interval: {}".format(confidence_interval))

        if training:
            score_file = "{}/scores.csv".format(self.name)
            # If file doesn't exist, add the header
            if not os.path.isfile(score_file):
                with open(score_file, "ab") as f:
                    f.write("avg_score,episode_count,frame_count,interval_lower,interval_upper,min,max\n")

            # Append scores to the file
            with open(score_file, "ab") as f:
                f.write("{},{},{},{},{},{},{}\n".format(avg_score, self.episode_count, self.frame_count, confidence_interval[0], confidence_interval[1], min(scores), max(scores)))

            count = 0
            for score in scores:
                if score >= 50:
                    count += 1
            if count >= len(scores) * 0.9:
                print("*** over 50 score in {} frames ***".format(self.frame_count))
                with open("pass_50.csv", "ab") as f:
                    f.write("{},{}\n".format(self.name, self.frame_count))
        else:
            with open("scores.txt", "ab") as f:
                for score in scores:
                    f.write("{},{}\n".format(self.name, score))
            


    def play(self):
        print("Playing {} agent after training for {} episodes or {} frames".format(self.name, self.episode_count, self.frame_count))
        reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}

        env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None, reward_values=reward_values)
        env.init()
        
        score = 0
        last_print = 0

        nb_episodes = 50
        while nb_episodes > 0:
            # pick an action
            state = env.game.getGameState()
            action = self.policy(state)                                          

            # step the environment
            reward = env.act(env.getActionSet()[action])

            score += reward

            # if reward == 1:``
            #     print(state)

            if score % 100 == 0 and score != last_print:
                print(int(score))
                last_print = score

            # reset the environment if the game is over
            if env.game_over():            
                # print("---------------")    
                # for s1, s2 in self.last_10:
                #     print(s1)
                #     print(s2)
                #     print("-=-=-")
                print("Score: {}".format(score))
                env.reset_game()
                nb_episodes -= 1
                score = 0