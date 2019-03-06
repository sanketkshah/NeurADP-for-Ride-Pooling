from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle

import pdb


MAX_EPISODE_LENGTH = 1440


def run(NUM_AGENTS=1000,
        EPISODE_LENGTH=MAX_EPISODE_LENGTH,
        START_HOUR=0,
        END_HOUR=9,
        IS_TRAINING=True):

    pdb.set_trace()

    # Initialising components
    envt = NYEnvironment()
    oracle = Oracle(envt)
    central_agent = CentralAgent()
    agents: List[LearningAgent] = []
    for agent_idx, initial_state in enumerate(envt.get_initial_state(NUM_AGENTS)):
        agent = LearningAgent(agent_idx, initial_state)
        agents.append(agent)

    # Iterating over episode
    assert(EPISODE_LENGTH <= MAX_EPISODE_LENGTH)
    request_generator = envt.get_request_batch(START_HOUR, END_HOUR)

    total_value_generated = 0
    num_total_requests = 0
    for _ in range(EPISODE_LENGTH):
        # Get new requests
        current_requests = next(request_generator)
        print("Current time: {}".format(envt.current_time))
        print("Number of new requests: {}".format(len(current_requests)))

        # Get feasible actions and score them
        print("Generating feasible requests...")
        scored_actions_all_agents = []
        for agent_idx in range(NUM_AGENTS):
            feasible_actions = oracle.get_feasible_actions(agents[agent_idx], current_requests)
            scored_actions = central_agent.get_value(agents[agent_idx], feasible_actions)
            scored_actions_all_agents.append(scored_actions)

        # Choose actions for each agent
        print("Choosing best actions...")
        final_actions = central_agent.choose_actions(scored_actions_all_agents)
        for agent_idx, action in enumerate(final_actions):
            agents[agent_idx].path = action.new_path

        # Get reward
        rewards = []
        for action in final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward
        print("Reward for epoch: {}".format(sum(rewards)))

        # Update value function
        if (IS_TRAINING):
            print("Updating value function...")
            central_agent.update_value_function(rewards, agents, final_actions)

        # Sanity check
        print("Sanity check...")
        for agent in agents:
            assert envt.has_valid_path(agent)

        # Simulate the passing of time
        print("Simulating motion till next epoch...")
        envt.simulate_motion(agents, current_requests)

        # Printing statistics for current epoch
        num_total_requests += len(current_requests)
        print('Number of requests accepted: {}'.format(total_value_generated))
        print('Number of requests seen: {}'.format(num_total_requests))
        print()

    return total_value_generated


if __name__ == '__main__':
    print(run())
