"""Simple reinforcement learning trading agent (actor-critic).

This is a minimal, educational Actor-Critic agent that trains on
historical data in an offline loop. It's a starting point for
experimentation — production trading agents require much more
robust environments, risk controls, and transaction cost modeling.
"""
import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple, Any, Dict
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


class TradingRLAgent:
    """A very small actor-critic agent that chooses actions {-1,0,1}.

    State: a flattened vector of the latest window of features.
    Action: -1 (short), 0 (flat), 1 (long).
    Reward: profit over next step (position * return) minus tiny cost.
    """

    def __init__(self, state_size: int = 60, action_size: int = 3, lr: float = 1e-4):
        self.state_size = state_size
        self.action_size = action_size

        # Actor
        state_in = Input(shape=(state_size,))
        a = Dense(128, activation='relu')(state_in)
        a = Dense(64, activation='relu')(a)
        probs = Dense(action_size, activation='softmax')(a)
        self.actor = Model(inputs=state_in, outputs=probs)
        self.actor.compile(optimizer=Adam(lr))

        # Critic
        c = Dense(128, activation='relu')(state_in)
        c = Dense(64, activation='relu')(c)
        value = Dense(1, activation='linear')(c)
        self.critic = Model(inputs=state_in, outputs=value)
        self.critic.compile(optimizer=Adam(lr), loss='mse')

        self.is_trained = False

    def _state_from_window(self, window: np.ndarray) -> np.ndarray:
        # Flatten the window into a vector
        return window.flatten()[None, :]

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        probs = self.actor.predict(state, verbose=0)[0]
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def train(self, df: pd.DataFrame, sequence_length: int = 60, episodes: int = 10):
        """Offline policy optimization over historical data.

        This is intentionally small and pedagogical.
        """
        try:
            feature_cols = [c for c in df.columns if c not in ['timestamp']]
            arr = df[feature_cols].values
            prices = df['close'].values

            for ep in range(episodes):
                # Slide over the dataset and perform small policy updates
                for t in range(sequence_length, len(arr) - 1):
                    window = arr[t-sequence_length:t]
                    state = self._state_from_window(window)
                    action = self.select_action(state)
                    # Map discrete action to position
                    position = action - 1  # 0->-1,1->0,2->1
                    reward = position * (prices[t+1] - prices[t]) - 0.0

                    # Critic update (value target = reward + V(next))
                    next_state = self._state_from_window(arr[t-sequence_length+1:t+1])
                    v_next = float(self.critic.predict(next_state, verbose=0)[0, 0])
                    target = reward + 0.99 * v_next
                    self.critic.train_on_batch(state, np.array([[target]]))

                    # Actor update via policy gradient with advantage
                    v = float(self.critic.predict(state, verbose=0)[0, 0])
                    advantage = target - v
                    # Construct a simple one-hot target weighted by advantage
                    probs = self.actor.predict(state, verbose=0)
                    action_onehot = np.zeros_like(probs)
                    action_onehot[0, action] = 1.0
                    # Multiply one-hot by advantage as a crude policy gradient signal
                    # We perform a custom gradient step using the actor's optimizer
                    with tf.GradientTape() as tape:
                        logits = self.actor(state, training=True)
                        logp = tf.math.log(tf.reduce_sum(logits * action_onehot, axis=1) + 1e-8)
                        loss = -logp * advantage
                    grads = tape.gradient(loss, self.actor.trainable_variables)
                    self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

            self.is_trained = True
            logger.info('RL trading agent trained (simple offline loop)')
            return True
        except Exception as e:
            logger.error(f'RL agent training failed: {e}')
            return False

    def act(self, observation: np.ndarray, deterministic: bool = False) -> int:
        st = observation.flatten()[None, :]
        return self.select_action(st, deterministic=deterministic)

    def get_model_info(self) -> Dict[str, Any]:
        return {'model_type': 'RL-Agent', 'is_trained': self.is_trained}
