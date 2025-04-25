This is a group project for Reinforcement Learning Course. We mainly tried to make three improvements based on the basic Blackjack-v1 environment：
1. Espand the action space by adding double-down, surrender and insurance, simulating complex gameplay strategies. (Code/Blackjack_actions_add)
2. Extended the environment’s state space by integrating a card-counting mechanism. We were able to simulate the situation that not shuffle deck each round. (Code/blackjack_keep_playing.ipynb & Code/blackjack_multiple_player.ipynb)
3. Since the first two setting were all trained by Q-learning, we tried use Monte Carlo in this part. (Codes/blackjack_MC.ipynb)
