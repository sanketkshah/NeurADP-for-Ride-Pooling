from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle
from ValueFunction import PathBasedNN, RewardPlusDelay, NeuralNetworkBased

from typing import List

import pdb
from copy import deepcopy
from itertools import repeat


def run_epoch(envt,
              oracle,
              central_agent,
              value_function,
              NUM_AGENTS,
              DAY,
              START_HOUR,
              END_HOUR,
              is_training,
              TRAINING_FREQUENCY: int=60):

    # Initialising agents
    agents: List[LearningAgent] = []
    initial_states = envt.get_initial_states(NUM_AGENTS, is_training)
    for agent_idx, initial_state in enumerate(initial_states):
        agent = LearningAgent(agent_idx, initial_state)
        agents.append(agent)

    # Iterating over episode
    request_generator = envt.get_request_batch(START_HOUR, END_HOUR, DAY)

    total_value_generated = 0
    num_total_requests = 0
    while True:
        # Get new requests
        try:
            current_requests = next(request_generator)
            print("Current time: {}".format(envt.current_time))
            print("Number of new requests: {}".format(len(current_requests)))
        except StopIteration:
            break

        # Get feasible actions
        feasible_actions_all_agents = []
        for agent_idx in range(NUM_AGENTS):
            feasible_actions = oracle.get_feasible_actions(agents[agent_idx], current_requests)
            feasible_actions_all_agents.append(feasible_actions)

        # Score feasible actions
        is_terminal = ((END_HOUR * 3600) - envt.EPOCH_LENGTH) == envt.current_time
        scored_actions_all_agents = value_function.get_value(agents, feasible_actions_all_agents, repeat(envt.current_time), repeat(is_terminal))

        # Choose actions for each agent
        final_actions = central_agent.choose_actions(scored_actions_all_agents, is_training=is_training, epoch_num=envt.num_days_trained)

        # Update
        if (is_training):
            # Update replay buffer
            value_function.remember(agents, feasible_actions_all_agents, is_terminal)

            # Update value function every TRAINING_FREQUENCY timesteps
            if ((int(envt.current_time) / int(envt.EPOCH_LENGTH)) % TRAINING_FREQUENCY == TRAINING_FREQUENCY - 1):
                value_function.update(central_agent)

        # Assign final actions to agents
        for agent_idx, (action, _) in enumerate(final_actions):
            agents[agent_idx].path = deepcopy(action.new_path)

        # Calculate reward for selected actions
        rewards = []
        for action, _ in final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward
        print("Reward for epoch: {}".format(sum(rewards)))

        # Sanity check
        for agent in agents:
            assert envt.has_valid_path(agent)

        # Simulate the passing of time
        envt.simulate_motion(agents, current_requests)
        num_total_requests += len(current_requests)

    # Printing statistics for current epoch
    print('Number of requests accepted: {}'.format(total_value_generated))
    print('Number of requests seen: {}'.format(num_total_requests))
    print()

    return total_value_generated


if __name__ == '__main__':
    # pdb.set_trace()

    # Constants
    NUM_AGENTS: int = 1000
    START_HOUR: int = 8
    END_HOUR: int = 9
    NUM_EPOCHS: int = 100
    TRAINING_DAYS: List[int] = list(range(3, 42))
    TEST_DAYS: List[int] = [2]
    SAVE_FREQ: int = 5

    # Initialising components
    envt = NYEnvironment(NUM_AGENTS, START_EPOCH=START_HOUR * 3600, STOP_EPOCH=END_HOUR * 3600)
    oracle = Oracle(envt)
    central_agent = CentralAgent()
    value_function = PathBasedNN(envt, load_model_loc='../models/PathBasedNN_10agent_94_279.h5')

    max_test_score = 0
    num_days_trained = 0
    for epoch_id in range(NUM_EPOCHS):
        for day in TRAINING_DAYS:
            envt.num_days_trained = num_days_trained
            total_requests_served = run_epoch(envt, oracle, central_agent, value_function, NUM_AGENTS, day, START_HOUR, END_HOUR, is_training=True)
            print("DAY: {}, Requests: {}".format(day, total_requests_served))
            value_function.tensorboard.on_epoch_end(num_days_trained, {'requests_served': total_requests_served})
            num_days_trained += 1

        test_score = 0
        for day in TEST_DAYS:
            total_requests_served = run_epoch(envt, oracle, central_agent, value_function, NUM_AGENTS, day, START_HOUR, END_HOUR, is_training=False)
            print("DAY: {}, Requests: {}".format(day, total_requests_served))
            test_score += total_requests_served

        # TODO: Save results better
        if (isinstance(value_function, NeuralNetworkBased)):
            if (test_score > max_test_score or (epoch_id % SAVE_FREQ) == (SAVE_FREQ - 1)):
                value_function.model.save('../models/{}_{}agent_{}_{}.h5'.format(type(value_function).__name__, NUM_AGENTS, epoch_id, test_score))
                max_test_score = test_score if test_score > max_test_score else max_test_score

    # pdb.set_trace()
