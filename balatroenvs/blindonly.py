import gymnasium as gym
import numpy as np
import itertools
from .env import *

class BlindEnv(BalatroEnvBase):
    def __init__(self):
        policy_states = ["select_cards_from_hand"]
        super().__init__(policy_states=policy_states)
        hand_size = 8
        max_cards = 5
        action_count = 0
        
        # Calculate the number of actions
        for i in range(1, max_cards+1):  # 1 to 5 cards
            # Play i cards
            action_count += len(list(itertools.combinations(range(1, hand_size + 1), i)))
        
        self.action_space= gym.spaces.Discrete(2 * action_count)
        # Define the observation space
        self.observation_space = gym.spaces.MultiBinary(n=8 * (13 + 4))

    def step(self, action, **kwargs):
         #super().step(action, **kwargs)
        action = self._process_action(action)
        if action[0] == Actions.DISCARD_HAND and self.state['discards_left'] == 0:
            action = [Actions.PLAY_HAND, [1,2,3,4,5]] # Invalid action, just play the hand
        
        lastchips = self.state['chips']
        lastround = self.state['round']

        info = self._step(action)
        #if self.success == False:
        #    return np.zeros(self.observation_space.shape), 0, True, True, {}      

        # Convert the observation to a more manageable format
        observation = self._process_observation(info)

        if action[0] == Actions.PLAY_HAND:
            roundchanged = observation['round'] != lastround
            newchips = observation['chips']
            if roundchanged:
                reward = newchips
            else:
                reward = newchips - lastchips
                if reward < 0:
                    reward = newchips # Chips can never go down so the model lost on the first round and started again  
            print("Played hand, reward: ", reward)
        else:
            reward = 0
            
        
        round_over = self.is_run_finished()

        
        # Convert hand to one-hot encoding
        obs = self.hand_vectorize(observation) 
        
        truncated = False 
        return obs, reward, round_over, truncated, info

    def hand_vectorize(self, observation):
        hand_one_hot = np.zeros((8, 13 + 4), dtype=np.int8)
        for i, card in enumerate(observation['hand']):
            if i < 8:  # Ensure we don't exceed 8 cards
                suit, rank = card.split('_')
                suit_index = ['C', 'D', 'H', 'S'].index(suit)
                rank_index = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K'].index(rank)
                hand_one_hot[i, rank_index] = 1  # One-hot encoding for rank
                hand_one_hot[i, 13 + suit_index] = 1  # One-hot encoding for suit
        return hand_one_hot.flatten()

    def reset(self, seed=None, options=None):
        #super().reset(**kwargs)
        info = self._reset()

        observation = self._process_observation(info)
        
        # Convert hand to one-hot encoding
        obs = self.hand_vectorize(observation)
        print(obs)
        return obs, info

    
    def _process_action(self, action_id):
        hand_size = 8
        max_cards = 5
        action_space = []
        
        parity = action_id % 2
        action_id = action_id // 2

        for i in range(1, max_cards+1):  # 1 to 5 cards
            action_space.extend(list(itertools.combinations(range(1, hand_size + 1), i)))
            
        action_type = Actions.PLAY_HAND if parity == 0 else Actions.DISCARD_HAND
        cards = action_space[action_id]
        return [action_type, cards]

    
    def _process_observation(self, observation):
        # Process the observation to extract relevant information
        # print(observation)
        state =  {
            'hand': [card['card_key'] for card in observation['hand']],
            'dollars': observation['game']['dollars'],
            'round': observation['game']['round'],
            'discards_left': observation['round']['discards_left'],
            'waiting_for': observation['waitingFor'],
            'chips': observation['game']['chips'],
            #'handscores': observation['handscores']
        }
        self.state = state
        return state

