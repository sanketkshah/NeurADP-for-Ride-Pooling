from Environment import NYEnvironment
from CentralAgent import CentralAgent
from LearningAgent import LearningAgent
from Oracle import Oracle


def run(NUM_VECHICLES=1000,
        EPISODE_LENGTH=1440,
        IS_TRAINING=True):

    envt = NYEnvironment()
    oracle = Oracle()
    central_agent = CentralAgent()
    agents = [LearningAgent() for _ in range(NUM_VECHICLES)]

    total_value_generated = 0
    for _ in range(EPISODE_LENGTH):
        current_requests = envt.get_reqeust_batch()

        feasible_actions_all_agents = []
        for agent_idx in range(NUM_VECHICLES):
            feasible_actions = oracle.get_feasible_actions(
                agents[agent_idx], current_requests)

            scored_actions = []
            for action in feasible_actions:
                scored_actions.append(
                    (action, agents[agent_idx].get_value(action)))

            feasible_actions_all_agents.append(scored_actions)

        final_actions = central_agent.choose_actions(
            feasible_actions_all_agents)

        rewards = []
        for action in final_actions:
            reward = envt.get_reward(action)
            rewards.append(reward)
            total_value_generated += reward

        if (IS_TRAINING):
            central_agent.update_value_function(rewards, agents, final_actions)

        for agent_idx, action in enumerate(final_actions):
            envt.update_state(agents[agent_idx], action)
            envt.simulate_motion(agents[agent_idx])

    return total_value_generated


if __name__ == '__main__':
    print(run())
