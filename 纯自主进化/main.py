from agent_module import Agent
import os
import agent_module

if __name__ == "__main__":
    # Set the proxy URL and port
    proxy_url = 'http://127.0.0.1'
    proxy_port = '7890'

    # Set the http_proxy and https_proxy environment variables
    os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
    os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

    key_path = "key.env"
    for _ in range(1):
        # goal_task = task_mgsm.MGSM_Task()
        goal_task_pth = "father_data"
        self_evolving_agent = Agent(goal_prompt_path="father_goal_prompt.md", key_path=key_path, own_task_pth=goal_task_pth, father_or_son=1, name="father")
        agent_module.agent_groups["begin-agent"] = {"object":self_evolving_agent,"descrption":"This is the initial agent, responsible for receiving tasks and assigning them."}
        self_evolving_agent.reinit()
        self_evolving_agent.evolve()

