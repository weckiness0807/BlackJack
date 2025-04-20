import os
from typing import Optional

import numpy as np
import gymnasium as gym  # ✅ Change `gym` to `gymnasium`
from gymnasium import spaces  # ✅ Import spaces from Gymnasium
from gymnasium.error import DependencyNotInstalled  # ✅ Import error handling from Gymnasium

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def draw_card(np_random):
    return int(np_random.choice(deck))

def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):
    return sum_hand(hand) > 21

def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)

def is_natural(hand):
    return sorted(hand) == [1, 10]

class BlackjackEnvWithInsurance(gym.Env):
    """Blackjack environment with Insurance added"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, natural=False, sab=False):
        # Updated action space: 0 = Stand, 1 = Hit, 2 = Double Down, 3 =Surrender, 4 =Insurance
        self.action_space = spaces.Discrete(5)  # ✅ Update to 5 actions

        # Observation space now includes (insurance available flag)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(32), spaces.Discrete(11), spaces.Discrete(2), spaces.Discrete(2))
        )

        self.natural = natural
        self.sab = sab
        self.render_mode = render_mode

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0.0
        terminated = False
        game_result = "Unknown"  # ✅ Initialize `game_result` early
    
        # ✅ Double Down (Action 2)
        if action == 2:
            if len(self.player) > 2:  # ❌ Double down only allowed on first two cards
                return self._get_obs(), -1, False, False, {
                    "num_cards": len(self.player), 
                    "dealer_hand": self.dealer, 
                    "game_result": "Invalid Move"
                }
    
            self.player.append(draw_card(self.np_random))  # Draw one final card
    
            if is_bust(self.player):  # Bust loses immediately
                reward = -2.0
                terminated = True
                game_result = "Loss"
            else:  # Otherwise, dealer plays
                terminated = True
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.np_random))
                reward = cmp(score(self.player), score(self.dealer)) * 2.0
                game_result = "Win" if reward > 0 else "Loss" if reward < 0 else "Push"
    
            return self._get_obs(), reward, terminated, False, {
                "num_cards": len(self.player),
                "dealer_hand": self.dealer,
                "game_result": game_result
            }
    
        # ✅ Surrender (Action 3)
        if action == 3:
            reward = -0.5  # Player loses half the bet
            terminated = True
            game_result = "Loss"
            return self._get_obs(), reward, terminated, False, {
                "num_cards": len(self.player),
                "dealer_hand": self.dealer,
                "game_result": game_result
            }
    
        # ✅ Insurance (Action 4)
        if action == 4:
            if self.dealer[0] != 1:  # ❌ Insurance only if dealer has an Ace
                reward -= 0.5  # ✅ Lose insurance bet
                game_result = "Loss"
                return self._get_obs(insurance_available=0), reward, False, False, {
                    "num_cards": len(self.player),
                    "dealer_hand": self.dealer,
                    "game_result": game_result
                }
    
            self.insurance_bet = 0.5  # ✅ Insurance is half the bet
    
            if is_natural(self.dealer):  
                reward = self.insurance_bet * 2  # ✅ Insurance pays 2:1 if dealer has Blackjack
                game_result = "Win"
                return self._get_obs(insurance_available=0), reward, True, False, {
                    "num_cards": len(self.player),
                    "dealer_hand": self.dealer,
                    "game_result": game_result
                }
            else:
                reward -= self.insurance_bet  # ✅ Lose insurance if dealer does not have Blackjack
                game_result = "Loss"
    
            return self._get_obs(insurance_available=0), reward, False, False, {
                "num_cards": len(self.player),
                "dealer_hand": self.dealer,
                "game_result": game_result
            }
    
        # ✅ Hit (Action 1)
        if action == 1:
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                terminated = True
                reward = -1.0
                game_result = "Loss"
            return self._get_obs(), reward, terminated, False, {
                "num_cards": len(self.player),
                "dealer_hand": self.dealer,
                "game_result": game_result
            }

        else:  # Stand (Dealer plays)
            terminated = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))

            reward = cmp(score(self.player), score(self.dealer))

            if self.sab and is_natural(self.player) and not is_natural(self.dealer):
                reward = 1.0  # Player wins automatically with a natural blackjack
            elif not self.sab and self.natural and is_natural(self.player) and reward == 1.0:
                reward = 1.5  # Natural blackjack payout

        if self.render_mode == "human":
            self.render()

        # ✅ Store dealer's final hand and game result in `info`
        game_result = "Win" if reward > 0 else "Loss" if reward < 0 else "Push"
        
        return self._get_obs(), reward, terminated, False, {
            "num_cards": len(self.player),
            "dealer_hand": self.dealer,  # ✅ Store dealer's full hand
            "game_result": game_result   # ✅ Store game outcome
        }
    
    def _get_obs(self, insurance_available=None):
        if insurance_available is None:  # If not manually overridden
            insurance_available = 1 if self.dealer[0] == 1 and len(self.player) <= 2 else 0 
            
        obs = (sum_hand(self.player), self.dealer[0], usable_ace(self.player), insurance_available)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        self.insurance_bet = 0  # Reset insurance bet

        obs = self._get_obs()  # ✅ Get observation
        
        # ✅ Create info dictionary
        info = {
            "num_cards": len(self.player)  # ✅ Store number of cards in player's hand
        }
        
        _, dealer_card_value, _, _ = self._get_obs()

        suits = ["C", "D", "H", "S"]
        self.dealer_top_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            self.dealer_top_card_value_str = "A"
        elif dealer_card_value == 10:
            self.dealer_top_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            self.dealer_top_card_value_str = str(dealer_card_value)

        if self.render_mode == "human":
            self.render()

        return obs, info  # ✅ Now returns both obs and info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

    def close(self):
        pass
