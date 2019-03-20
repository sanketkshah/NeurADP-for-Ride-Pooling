from LearningAgent import LearningAgent
from Action import Action
from Environment import Environment

from typing import List, Tuple, Deque, Dict, Any

from abc import ABCMeta, abstractmethod
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Masking, Concatenate, Flatten
from keras.models import Model
from random import sample
from collections import deque
import numpy as np
from itertools import repeat


class ValueFunction(metaclass=ABCMeta):
    """docstring for ValueFunction"""

    def __init__(self):
        super(ValueFunction, self).__init__()

    @abstractmethod
    def get_value(self, agent: LearningAgent, feasible_actions: List[Action], current_time: float) -> List[Tuple[Action, float]]:
        raise NotImplementedError

    @abstractmethod
    def update(self, agents: List[LearningAgent], final_actions: List[Action], candidate_actions: List[List[Action]], is_terminal: bool):
        raise NotImplementedError


class ImmediateReward(ValueFunction):
    """docstring for ImmediateReward"""

    def __init__(self):
        super(ImmediateReward, self).__init__()

    def get_value(self, agent: LearningAgent, feasible_actions: List[Action], current_time: float) -> List[Tuple[Action, float]]:
        scored_actions: List[Tuple[Action, float]] = []
        for action in feasible_actions:
            score = sum([request.value for request in action.requests])
            scored_actions.append((action, score))

        return scored_actions

    def update(self, *args, **kwargs):
        pass


class RewardPlusDelay(ValueFunction):
    """docstring for RewardPlusDelay"""

    def __init__(self, DELAY_COEFFICIENT: float=10e-4):
        super(RewardPlusDelay, self).__init__()
        self.DELAY_COEFFICIENT = DELAY_COEFFICIENT

    def get_value(self, agent: LearningAgent, feasible_actions: List[Action], current_time: float) -> List[Tuple[Action, float]]:
        scored_actions: List[Tuple[Action, float]] = []
        for action in feasible_actions:
            assert action.new_path

            immediate_reward = sum([request.value for request in action.requests])
            remaining_delay_bonus = self.DELAY_COEFFICIENT * action.new_path.total_delay
            score = immediate_reward + remaining_delay_bonus

            scored_actions.append((action, score))

        return scored_actions

    def update(self, *args, **kwargs):
        pass


class NeuralNetworkBased(ValueFunction, metaclass=ABCMeta):
    """docstring for NeuralNetwork"""

    def __init__(self, envt: Environment, GAMMA: float=0.9, BATCH_SIZE: int=1024):
        super(NeuralNetworkBased, self).__init__()

        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.envt: Environment = envt

        self.prev_state: List[Tuple[float, LearningAgent, Action]] = []
        self.replay_buffer: Deque[Tuple[float, LearningAgent, Action, List[Action], bool]] = deque(maxlen=500000)

        # Get NN Model
        self.model = self._init_NN(self.envt.NUM_LOCATIONS)

        # Define Loss and Compile
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    @abstractmethod
    def _init_NN(self, num_locs: int):
        raise NotImplementedError()

    @abstractmethod
    def _format_input(self, feasible_actions: List[Tuple[float, LearningAgent, Action]]):
        raise NotImplementedError()

    def get_value(self, agent: LearningAgent, feasible_actions: List[Action], current_time: float) -> List[Tuple[Action, float]]:
        action_inputs = self._format_input([(current_time, agent, action) for action in feasible_actions])
        expected_future_values = self.model.predict(action_inputs, batch_size=self.BATCH_SIZE)

        scores = [float(self.envt.get_reward(action) + self.GAMMA * value) for action, value in zip(feasible_actions, expected_future_values)]

        return list(zip(feasible_actions, scores))

    def _update_replay_buffer(self, candidate_actions_all_agents: List[List[Action]], is_terminal: bool):
        # TODO: Prioritised Experience Replay
        for prev_state, candidate_next_actions in zip(self.prev_state, candidate_actions_all_agents):
            self.replay_buffer.append((prev_state[0], prev_state[1], prev_state[2], candidate_next_actions, is_terminal))

    def update(self, agents: List[LearningAgent], final_actions: List[Action], candidate_actions: List[List[Action]], is_terminal: bool):
        # Update replay buffer
        if (self.prev_state):
            self._update_replay_buffer(candidate_actions, is_terminal)

            # Sample from replay buffer and update Q-Network
            num_samples = min(10000, len(self.replay_buffer))
            experiences = sample(self.replay_buffer, num_samples)

            # Find TD-Target
            supervised_targets: List[float] = []
            for experience in experiences:
                current_time, agent, current_action, possible_next_actions, is_terminal = experience
                if (is_terminal):
                    values_next_state = 0.0
                else:
                    time_at_next_epoch = current_time + self.envt.EPOCH_LENGTH
                    values_next_state = max(score for _, score in self.get_value(agent, possible_next_actions, time_at_next_epoch))
                supervised_targets.append(values_next_state)

            # Update based on TD-Target
            action_input = self._format_input([(time, agent, action) for time, agent, action, _, _ in experiences])
            self.model.fit(action_input, supervised_targets, batch_size=self.BATCH_SIZE)

        self.prev_state = list(zip(repeat(self.envt.current_time), agents, final_actions))


class PathBasedNN(NeuralNetworkBased):

    def __init__(self, envt: Environment):
        super(PathBasedNN, self).__init__(envt)

    def _init_NN(self, num_locs: int):
        # DEFINE NETWORK STRUCTURE
        # Get path and current locations' embeddings
        path_location_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1,), dtype='int32', name='path_location_input')
        location_embed = Embedding(output_dim=10, input_dim=self.envt.NUM_LOCATIONS + 1, mask_zero=True, name='location_embedding')
        path_location_embed = location_embed(path_location_input)

        # Get associated delay for different path locations
        delay_input = Input(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1), name='delay_input')
        delay_embed = TimeDistributed(Dense(10, activation='relu', name='delay_embedding'))(delay_input)

        # Get entire path's embedding
        path_input = Concatenate()([path_location_embed, delay_embed])
        path_embed = LSTM(100)(path_input)

        # Get current time's embedding
        current_time_input = Input(shape=(1,), name='current_time_input')
        current_time_embed = Dense(10, activation='relu', name='time_embedding')(current_time_input)

        # Get Embedding for the entire thing
        state_embed = Concatenate()([path_embed, current_time_embed])
        state_embed = Dense(100, activation='relu', name='state_embed_1')(state_embed)
        state_embed = Dense(100, activation='relu', name='state_embed_2')(state_embed)
        state_embed = Dense(100, activation='relu', name='state_embed_3')(state_embed)

        # Get predicted Value Function
        output = Dense(1)(state_embed)

        model = Model(inputs=[path_location_input, delay_input, current_time_input], outputs=output)

        return model

    def _format_input(self, feasible_actions: List[Tuple[float, LearningAgent, Action]]):
        input: Dict[str, List[Any]] = {"path_location_input": [], "delay_input": [], "current_time_input": []}

        for current_time, agent, action in feasible_actions:
            input["current_time_input"].append(current_time / 86400)

            location_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1,))
            delay_order: np.ndarray = np.zeros(shape=(self.envt.MAX_CAPACITY * 2 + 1, 1)) - 1

            # Adding current location
            location_order[0] = agent.position.next_location + 1

            assert action.new_path is not None
            for idx, node in enumerate(action.new_path.request_order):
                if (idx >= 20):
                    break

                location, deadline = action.new_path.get_info(node)
                visit_time = node.expected_visit_time

                location_order[idx + 1] = location + 1
                delay_order[idx + 1, 0] = (deadline - visit_time) / 6000  # normalising

            input["path_location_input"].append(location_order)
            input["delay_input"].append(delay_order)

        # Convert to numpy array
        for key, value in input.items():
            input[key] = np.array(value)

        return input


class SimpleNN(NeuralNetworkBased):

    def __init__(self, envt: Environment):
        super(SimpleNN, self).__init__(envt)

    def _init_NN(self, num_locs: int) -> Model:
        # TODO: Add information about the action

        # DEFINE NETWORK STRUCTURE
        # Get current and next locations' embedding
        location_input = Input(shape=(2,), name='location_input')
        location_embed = Embedding(output_dim=10, input_dim=self.envt.NUM_LOCATIONS, input_length=2, name='location_embedding')
        current_location_embed = location_embed(location_input)

        # Get current time and delay
        time_input = Input(shape=(2,), name='time_input')

        # Put them together
        state_features = Concatenate()([Flatten()(current_location_embed), time_input])

        # Add dense layers
        state_embed = state_features
        NUM_DENSE_LAYERS = 3
        for _ in range(NUM_DENSE_LAYERS):
            state_embed = Dense(100, activation='relu')(state_embed)

        # Get predicted Value Function
        output = Dense(1)(state_embed)

        model = Model(inputs=[location_input, time_input], outputs=output)

        return model

    def _format_input(self, feasible_actions: List[Tuple[float, LearningAgent, Action]]):
        input: Dict[str, List[Any]] = {"location_input": [], "time_input": []}

        for current_time, agent, action in feasible_actions:
            current_location = agent.position.next_location
            assert action.new_path is not None

            next_location = action.new_path.get_next_location()
            next_location = next_location if next_location != -1 else current_location
            input["location_input"].append([current_location, next_location])

            remaining_delay = action.new_path.total_delay
            input["time_input"].append([current_time / 86400, remaining_delay / 6000])  # normalising

        # Convert to numpy array
        for key, value in input.items():
            input[key] = np.array(value)

        return input
