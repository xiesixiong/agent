import os
import io
import re
import ast
import sys
import json
import typing
import inspect
import functools
import itertools
import traceback
import importlib
import subprocess
import contextlib
import collections
import openai
import logic
import task_mgsm
import threading
from assist import CheckAssistant
from datetime import datetime
# #Generate a counter dictionary: By default, all dictionary values that do not exist are int 0
# agent.action_counter = collections.defaultdict(int)



def read_task(task_pth):
    task = []
    with open(task_pth,'r',encoding="utf-8") as f:
        for t in f :
            task.append(t)
    return task



#Write a base class Agent to define some abstract functions
class AgentBase:

    def execute_action(agent,*args,**kwargs):
        raise NotImplementedError("The Agent abstract class function execute_action has not been implemented yet!")

    def action_call_llm(agent,*args,**kwargs):
        raise NotImplementedError("The Agent abstract class function action_call_llm has not been implemented yet!")



class Agent:

    def __init__(agent,api_key = None,goal_prompt_path = 'father_goal_prompt.md',key_path = 'key.env',own_task_pth = None,father_or_son = None,name = None):

        agent.goal_prompt = open(goal_prompt_path, 'r').read()
        agent.own_task_pth = own_task_pth
        if api_key is None:
            api_key = open(key_path, 'r').read().strip()
        openai.api_key = api_key
        agent.api_key = api_key

        agent.client = openai.OpenAI(api_key=api_key)
        agent.son_agents = {}
        agent.son_tasks = []
        agent.son_results = []
        agent.father_or_son = father_or_son
        agent.answer = None
        agent.name = name
        agent.action_counter = collections.defaultdict(int)
        with open(f"{name}的日志",'w',encoding="utf-8") as f:
            pass

        agent.original_solver = agent.solver
        agent.best_solver = agent.original_solver
        agent.best_perfomance = float('-inf')
        # Initialize optimization history and iterations

        agent.action_functions = [
            #第一个功能：展示出来分析的内容，分析包括：任务资源、反馈、下一步的计划
            {
                "type": "function",
                "function": {
                    "name": "action_display_analysis",
                    "description": "Display an analysis of the current state, including available resources, logic (of solver or other actions) and evaluation feedbacks from the target task,and decide whether and how to break the problem down into detailed son-problems,and you can generate some self-derivatives to help you solve sub-problems, and reasons or plans for the next actions based on this analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis": {
                                "type": "string",
                                "description": "A detailed analysis of the current state, including reasons or plans for the following actions."
                            }
                        },
                        "required": ["analysis"],
                        "additionalProperties": False,
                    },
                    "strict": True
                }
            },

            #第二个功能：感受环境，包括全局资源，一切可以用到的东西
            {
                "type": "function",
                "function": {
                    "name": "action_environment_aware",
                    "description": "Reflect and summarize available resources of the current runtime environment including variables, functions, modules, and external libraries.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },

            #第三个功能：读取一切代码，指定模块中指定的函数
            {
                "type": "function",
                "function": {
                    "name": "action_read_logic",
                    "description": "Reads the source code of the specified logic (function, method, or class) within a given module.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to read. If the target_name contains a dot, it refers to a method within a class (e.g., 'Agent.action_call_llm')."
                            }
                        },
                        "required": ["module_name", "target_name"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },

            #第四个功能：修改代码，增加、删除、修改 指定模块指定的函数的代码
            {
                "type": "function",
                "function": {
                    "name": "action_adjust_logic",
                    "description": "Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "module_name": {
                                "type": "string",
                                "description": "The module where the logic resides."
                            },
                            "target_name": {
                                "type": "string",
                                "description": "The name of the function, method, or class to modify/add/delete. If the target_name contains a dot, it refers to a method within a class."
                            },
                            "new_code": {
                                "type": "string",
                                "description": "The new logic as a string. (Ensure there is no extra indentation in new_code)"
                            },
                            "target_type": {
                                "type": "string",
                                "enum": ["function", "class"],
                                "description": "The type of target."
                            },
                            "operation": {
                                "type": "string",
                                "enum": ["modify", "add", "delete"],
                                "description": "The operation to perform."
                            }
                        },
                        "required": ["module_name", "target_name", "new_code", "target_type", "operation"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },

            #第五个功能：运行代码，具备执行某个代码文件的能力
            {
                "type": "function",
                "function": {
                    "name": "action_run_code",
                    "description": "Execute Python or shell code and capture the output, errors, and return value. (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code_type": {
                                "type": "string",
                                "enum": ["python", "bash"],
                                "description": "The type of code to execute."
                            },
                            "code": {
                                "type": "string",
                                "description": "The code to execute as a string."
                            }
                        },
                        "required": ["code_type", "code"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },

            #第六个功能：在指定任务上执行，并得到任务集合的正确率等结果反馈
            {
                "type": "function",
                "function": {
                    "name": "action_evaluate_on_task",
                    "description": "Evaluate the current solver on the goal task samples and return the evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },

            #第七个功能：调用外部大模型来帮助它完成任务，返回为json格式
            {
                "type": "function",
                "function": {
                    "name": "action_call_json_format_llm",
                    "description": "Call an external LLM for assistance with gathering insights, refining strategies, correcting errors, and solving complex problems. Output response in JSON format.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model": {
                                "enum": ["gpt-4o-mini", "gpt-4o"],
                                "description": "ID of the model to use."
                            },
                            "messages": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "role": {
                                            "enum": ["system", "assistant", "user"]
                                        },
                                        "content": {"type": "string"}
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False
                                },
                                "description": "A list of messages comprising the conversation so far."
                            },
                            "temperature": {
                                "type": "number",
                                "description": "What sampling temperature to use. Higher values will make the output more random, while lower values will make it more focused and deterministic."
                            },
                            "role": {
                                "type": "string",
                                "description": "The role that LLM play."
                            },
                            "return_dict_keys": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "An array containing the names of the keys that should be present in the returned dictionary."
                            },
                            "requirements": {
                                "type": "string",
                                "description": "A string that specifies the conditions required to perform a call to the LLM."
                            }
                        },
                        "required": ["model", "messages", "temperature", "role", "return_dict_keys", "requirements"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
        #第九个功能：让助手启动功能
            {
                "type": "function",
                "function": {
                    "name": "create_son_agents",
                    "description": "Create son-agents to address the corresponding son-problems.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_description": {
                                "type": "string",
                                "description": "Description of the task you want the son_agent to complete."
                            },
                            "son_agent_name": {
                                "type": "string",
                                "description": "The name of this son_agent should be named."
                            }
                        },
                        "required": [ "task_description", "son_agent_name"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },

            # 第十个功能:得到子agent的答案
            {
                "type": "function",
                "function": {
                    "name": "call_son_agents",
                    "description": "After the son-agents have evolved for a while, the father agent collects their optimal solutions to their respective son-problems.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "son_agent_name": {
                                "type": "string",
                                "description": "This name should refer to the designated son_agent to perform this task."
                            },
                            "specific_questions": {
                                "type": "string",
                                "description": "This specific_questions description  about what you need this son_agent to help you solve."
                            }
                        },
                        "required": [ "son_agent_name","specific_questions"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },



        ]

        #记录优化的历史过程
        agent.optimize_history = []

        agent.check_assistant = None



    def create_son_agents(agent,task_description,son_agent_name):
        if agent.father_or_son ==1:
            #先创建子agent的数据集
            son_data = []
            own_task = read_task(agent.own_task_pth)
            for task in own_task:
                completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"你帮我把这个问题{task}按照{task_description}进行改写，得到一个新的问题"
                        },
                    ],
                )

                son_data.append(completion.choices[0].message.content)

            with open(f'{son_agent_name}dataset.txt', 'w',encoding="utf-8") as file:
                for item in son_data:
                    file.write(item + '\n')

            son_agent = Agent(api_key=agent.api_key,goal_prompt_path="son_goal_prompt.md",own_task_pth=f'{son_agent_name}dataset.txt',father_or_son=0,name=son_agent_name)
            agent.son_agents[son_agent_name] = {"object":son_agent,"description":f"This son_agent is used to solve {task_description} "}
            thread = threading.Thread(target=son_agent.evolve)
            thread.start()

            with open(f"{agent.name}的日志", 'a', encoding="utf-8") as f:
                # 获取当前时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 写入日志信息，分为多行以便更易于阅读
                f.write(f"[{timestamp}] 当前Agent: {agent.name}\n")
                f.write(f"  - 正在调用函数: create_son_agents\n")
                f.write(f"  - 任务描述: {task_description}\n")
                f.write(f"  - 子Agent名字: {son_agent_name}\n")
                f.write(f"  - 子数据集已创建，且子Agent已开始运行!\n")
                f.write("\n\n")  # 添加空行分隔日志条目

            print(( f"当前agent是{agent.name}正在调用函数create_son_agents，创建针对于{task_description}的子数据集，并命名子agent名字为{son_agent_name}，子数据集已创建好！为了解决{task_description}的agent已开始运行！\n\n"))
            return "create_son_agents is ok!"
        return "you don't be allowed"



    # 把对应的任务输入给对应的子agent，得到结果
    def call_son_agents(agent,son_agent_name,specific_questions):
        if agent.father_or_son == 1:
            son_agent = agent.son_agents[son_agent_name]["object"]
            answer = son_agent.best_solver(specific_questions) # todo: son_agent.archived_solver; son_agent.archived_score
            agent.son_results.append(f"为了解决{specific_questions}这个子任务得到的方案是：{answer}")

            with open(f"{agent.name}的日志", 'a', encoding="utf-8") as f:
                # 获取当前时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 写入日志信息，分为多行以便更易于阅读
                f.write(f"[{timestamp}] 当前Agent: {agent.name}\n")
                f.write(f"  - 正在调用函数: call_son_agents\n")
                f.write(f"  - 被调用的子Agent: {son_agent_name}\n")
                f.write(f"  - 子Agent处理的任务: {specific_questions}\n")
                f.write(f"  - 返回的结果: {answer}\n")
                f.write("\n\n")  # 添加空行分隔日志条目



            print((f"当前agent是{agent.name},他正在调用函数  call_son_agents ，call的是子agent{son_agent_name}，当前子agent处理的任务是{specific_questions}，返回的结果是{answer} \n\n"))
            return f"当前agent是{agent.name}",f"当前子agent处理的任务是{specific_questions}，返回的结果是{answer}"
        return "you don't be allowed"





    #基础功能，调用外界大模型
    def action_call_json_format_llm(
            agent,
            *,
            messages: typing.List[typing.Dict[str, str]],
            model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini",
            temperature: float = 1.0,
            max_completion_tokens: int = 4096,
            num_of_response: int = 1,
            role: str = "task solver",
            return_dict_keys: typing.List[str] = [],
            requirements: str = "",
    ):
        system_prompt = (
            f"You are a helpful {role}.\n"
            f"Reply in JSON format, ONLY using the keys {return_dict_keys}.\n"
            f"Requirements:\n{requirements}"
        ).strip()
        _messages = [{"role": "system", "content": system_prompt}, *messages]
        return_dicts = agent.action_call_llm(model=model,
                                             messages=_messages,
                                             temperature=temperature,
                                             max_completion_tokens=max_completion_tokens,
                                             n=num_of_response,
                                             response_format="json")

        for key in return_dict_keys:
            for return_dict in return_dicts:
                if key not in return_dict:
                    return_dict[key] = f"NO {key} IN DICTIONARY"
        return return_dicts

    def action_call_llm(
            agent,
            *,
            model: typing.Literal["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"] = "gpt-4o",
            messages: typing.List[typing.Dict[str, str]],
            temperature: float = 1.0,
            max_completion_tokens: int = 4096,
            n: int = 1,
            response_format: typing.Literal["text", "json", "json_object"] = "text",
            tools=None,
            tool_choice=None,
    ):
        """
        Sends a request to the OpenAI LLM with a system prompt and user message, and returns the response.

        Args:
            agent (Agent): The OpenAI client instance used to interact with the LLM.
            messages (List[Dict[str, str]]): A list of message dictionaries (conversation history).
            response_format (str): The desired format of the LLM's output.
            model (str): Specifies which LLM model to use.
            temperature (float): A float value controlling the randomness of the model's responses. Higher values (e.g., 1.0) increase creativity, while lower values (e.g., 0.1) make the responses more focused and deterministic.
            max_completion_tokens: An integer defining the maximum number of tokens in the completion response, up to 4096.
            n (int): The number of chat completion choices to generate for each input message.

        Returns:
            response (dict): The response from the OpenAI LLM.
        """
        try:
            if response_format == "json":
                response_format = "json_object"

            import copy
            messages = copy.deepcopy(messages)
            for message in messages:
                message["content"] = str(message["content"])

            kwargs = {
                "n": n,
                "model": model,
                "messages": messages,
                "response_format": {"type": response_format if response_format == "json_object" else "text"},
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens
            }

            if tools is not None:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = agent.client.chat.completions.create(**kwargs).to_dict()  # to Python dictionary

            def try_parse_json(content):
                try:
                    return json.loads(content)
                except:
                    return {"JSONDecodeError": content}

            if response_format == "text":
                return [choice["message"] for choice in response["choices"]]
            else:
                return [try_parse_json(choice["message"]["content"]) for choice in response["choices"]]
        except Exception as e:
            raise e


    #对agent对象进行重新初始化，更新一些属性，尤其是optimize-histroy，记录当前的solver代码
    def reinit(agent):
        agent.optimize_history = []
        first_aware_content = action_environment_aware(agent)
        solver_logic = action_read_logic("agent_module", "Agent.solver")


        print(f"当前agent是{agent.name}正在对环境进行分析",first_aware_content, end="\n\n")
        print(f"当前agent是{agent.name}打印自己的solver逻辑",solver_logic, end="\n\n")

        #将当前的solver代码存储到自己的历史记忆中
        # agent.optimize_history.append({"role": "user", "content": first_aware_content})
        agent.optimize_history.append({"role": "user", "content": "The logic of solver:\n" + solver_logic})


    def execute_action(agent,actions:typing.Dict):

        #先记录一下是否需要重新初始化一下
        is_reinit = False

        #遍历传入的actions，来执行
        for tool_call in actions['tool_calls']:

            with open(f"{agent.name}的日志", 'a', encoding="utf-8") as f:
                # 获取当前时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 写入日志信息，分为多行
                f.write(f"[{timestamp}] 当前Agent: {agent.name}\n")
                f.write(f"  - 正在尝试调用工具: {tool_call}\n")
                f.write("\n\n")  # 添加空行分隔日志条目

            print(f"当前agent是{agent.name}","当前agent正在尝试调用工具（函数）：",tool_call,end="\n\n")

            try:
                agent.action_counter[tool_call['function']['name']] += 1
                arguments = json.loads(tool_call['function']['arguments']) if tool_call['function']['arguments'] else {}
                if tool_call['function']['name'] == "action_display_analysis":
                    result = action_display_analysis(**arguments)

                elif tool_call['function']['name'] == "action_environment_aware":
                    result = action_environment_aware(agent, **arguments)

                elif tool_call['function']['name'] == "action_read_logic":
                    result = action_read_logic(**arguments)

                elif tool_call['function']['name'] == "action_adjust_logic":
                    result = action_adjust_logic(**arguments)
                    print("此时调整完代码了观察一下solver")
                    print(action_read_logic("agent_module","Agent.solver"))

                elif tool_call['function']['name'] == "action_run_code":
                    result = action_run_code(**arguments)
                    if arguments.get("code_type", None) == "python" and "self_evolving_agent.reinit()" in arguments.get(
                            "code", ""):
                        is_reinit = True
                elif tool_call['function']['name'] == "action_call_llm":
                    result = agent.action_call_llm(**arguments)


                # elif tool_call['function']['name'] == "split_task":
                #     result = agent.split_task()
                #     print(f"当前agent是{agent.name}","任务已经分割好")

                elif tool_call['function']['name'] == "create_son_agents":
                    result = agent.create_son_agents(**arguments)


                elif tool_call['function']['name'] == "call_son_agents":
                    result = agent.call_son_agents(**arguments)


                elif tool_call['function']['name'] == 'action_call_json_format_llm':
                    result = agent.action_call_json_format_llm(**arguments)
                    try:
                        print(f"当前agent是{agent.name}",json.loads(result[0]))
                    except:
                        print(f"当前agent是{agent.name}",result[0])

                elif tool_call['function']['name'] == "action_evaluate_on_task":
                    result, score = action_evaluate_on_task(agent.own_task_pth,agent.best_solver)

                    if score >= agent.best_perfomance:
                        agent.best_perfomance = score
                        agent.best_solver = agent.solver
                        agent.save_solver(agent.best_solver)
                        print("新solver保存好了")
                        print("此时的solver是这个样子的")
                        print(action_read_logic("agent_module","Agent.solver"))


                else:
                    raise ValueError(f"Unknown function name: {tool_call['function']['name']}")

            except Exception as e:
                agent.action_counter["error_handle"] += 1
                exception_stringio = io.StringIO()
                traceback.print_exc(file=exception_stringio)
                result = "Error " + exception_stringio.getvalue()
                exception_stringio.close()

            with open(f"{agent.name}的日志", 'a', encoding="utf-8") as f:
                # 获取当前时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 写入日志信息，格式化更清晰
                f.write(f"[{timestamp}] 当前Agent: {agent.name}\n")
                f.write(f"  - 工具调用结果:\n")
                f.write(f"    {result}\n")
                f.write("\n\n")  # 添加空行分隔日志条目

            print(f"当前agent是{agent.name}","tool call result:\n", result, sep="", end="\n\n")
            if is_reinit:
                break
            agent.optimize_history.append({"role": "tool",
                                           "content": result,
                                            "tool_call_id": tool_call['id']})

        with open(f"{agent.name}的日志", 'a', encoding="utf-8") as f:
            # 获取当前时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 写入日志信息，分为多行以便更易于阅读
            f.write(f"[{timestamp}] 当前Agent: {agent.name}\n")
            f.write(f"  - Action Counter: {agent.action_counter}\n")
            f.write("\n\n")  # 添加空行分隔日志条目

        print(f"当前agent是{agent.name}","Action Counter:", agent.action_counter, end='\n\n')
        if agent.action_counter["evolve"] >= 30:
            sys.exit(1)
        with open(f"{agent.name}的日志", 'a',encoding="utf-8") as f:
            f.write(f"当前agent是{agent.name}执行完tool的调用了 要开始下一个 Agent Evolve \n\n ")
        print(f"当前agent是{agent.name}要开始","Agent Evolve", end="\n\n")

        agent.evolve()

    def save_solver(agent, solver_func):
        # 保存函数代码到文件
        solver_code = inspect.getsource(solver_func)
        with open('best_solver.py', 'a') as f:
            f.write(solver_code)
            f.write('\n\n')
            f.write(f"Saved new best solver with performance:{agent.best_perfomance}")
            f.write('\n\n')



    def solver(agent, task: str):
        messages = [{"role": "user", "content": f"# Your Task:\n{task}"}]
        response = agent.action_call_json_format_llm(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.8,
            num_of_response=1,
            role="Travel Planner",
            return_dict_keys=["reasoning", "answer"],
            requirements=(
                "1. Please give me a detailed and reasonable plan.\n"
            ).strip(),
        )

        return_dict = response[0]
        return_dict["answer"] = str(return_dict.get("answer", ""))
        return return_dict

    #执行进化迭代，通过与llm交互，得到改进建议，然后自身迭代
    def evolve(agent):
        with open(f"{agent.name}的日志", 'a',encoding="utf-8") as f:
            f.write(f"当前agent是{agent.name}自身尝试进化中》》》》》》》》》》》》 \n\n ")
        print(f"当前agent是{agent.name}","自身尝试进化中》》》》》》》》》》》》")
        agent.action_counter["evolve"] += 1

        tool_call_ids = set()
        remain_optimize_history = []
        # print(f"当前agent是{agent.name}","当前的历史信息里是>>>>>",agent.optimize_history,"\n\n")
        for message in agent.optimize_history[-10:]:
            if message["role"] == "assistant" and message["tool_calls"]:
                tool_call_ids = set()
                for tool_call in message["tool_calls"]:
                    tool_call_ids.add(tool_call["id"])
            if message["role"] == "tool" and message["tool_call_id"] not in tool_call_ids:
                print(f"当前agent是{agent.name}",f"pop item: {message}", end='\n\n')
                continue
            remain_optimize_history.append(message)
        agent.optimize_history = remain_optimize_history

        messages = [{"role": "system", "name": "Principles", "content": agent.goal_prompt},
                    {"role": "system", "name": "Environment", "content": action_environment_aware(agent)},
                    *agent.optimize_history]
        try:
            response = agent.action_call_llm(messages=messages, model="gpt-4o", response_format="text",
                                             tools=agent.action_functions, tool_choice="required")

        except Exception as e:
            print(f"当前agent是{agent.name}",repr(e))
            for message in messages:
                print(f"当前agent是{agent.name}",message)
            sys.exit(1)

        agent.optimize_history.append(response[0])
        # print(f"当前agent是{agent.name}","执行完一次evolve后当前的历史信息里是>>>>>", agent.optimize_history, "\n\n")
        with open(f"{agent.name}的日志", 'a', encoding="utf-8") as f:
            # 获取当前时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # 写入日志信息，分为多行以便更易于阅读
            f.write(f"[{timestamp}] 当前Agent: {agent.name}\n")
            f.write(f"  - 完成elove任务\n")
            f.write(f"  - 准备执行action: {response[0]}\n")
            f.write("\n")  # 添加空行分隔日志条目

        print(f"当前agent是{agent.name}","开始准备执行exection-action",response[0],"\n\n")
        agent.execute_action(response[0])












#功能一：展示获取到的分析内容 到控制台上 同时返回指令告诉agent已经接受到analysis
def action_display_analysis(analysis):

    print("当前的分析内容为：",analysis,"\n\n")
    return "Analysis Received. Just do it!"


#功能二：分析当前运行环境一切资源，包括 全局变量、函数、模块、类 以及 agent对象的方法和属性
def action_environment_aware(agent: AgentBase):

    #这个内部函数是为了记录各个资源的情况
    def summarize_items(items, header):
        summary = [header]
        for name, value in items:
            if not name.startswith('__'):
                if name in ['goal_prompt']:
                    summary.append(f"- {name} = Your {name}.")
                elif name in ['optimize_history', 'function_map', 'action_functions']:
                    summary.append(f"- {name} = The length of your {name} is {len(getattr(agent, name))}.")
                elif name in ['logic']:
                    pass
                else:
                    summary.append(f"- {name} = {value}")
        if len(summary) == 1:
            summary.append("- None")
        return summary

    summary = []

#这一部分是获取全局的资源信息：
    #这一块是记录全局资源，包括：全局函数。全局类，全局模块，普通变量
    global_vars = [(k, v) for k, v in globals().items() if not k.startswith('__') and k != "AgentBase"]
    functions = [(k, v) for k, v in global_vars if inspect.isfunction(v)]
    calsses = [(k, v) for k, v in global_vars if inspect.isclass(v)]
    modules = [(k, v) for k, v in global_vars if inspect.ismodule(v)]
    variables = [(k, v) for k, v in global_vars if
                 not (inspect.isfunction(v) or inspect.isclass(v) or inspect.ismodule(v))]
    #把这些信息都存到summary中
    summary.extend(summarize_items(functions, "\nGlobal Functions:"))
    summary.extend(summarize_items(modules, "\nGlobal Modules:"))
    summary.extend(summarize_items(variables, "\nGlobal Variables:"))
    summary.extend(summarize_items(calsses, "\nGlobal Calsses:"))

#这一部分是获取agent对象自己的属性和功能信息：
    methods = inspect.getmembers(agent, inspect.ismethod)
    attributes = inspect.getmembers(agent, lambda x: not inspect.ismethod(x))

    summary.extend(summarize_items(methods, "\nCurrent Agent Instance's Methods:"))
    summary.extend(summarize_items(attributes, "\nCurrent Agent Instance's Attributes:"))

    return "\n".join(summary).strip()


#功能三：读取指定模块中的源代码，包括模块中的函数、方法或类对象  传入参数为：模块的名字module-name ，这个模块下你想访问的资源（类对象，方法...）target-name 就是一切你想知道的源代码都可以知道
#最终以字符串的形式保存
def action_read_logic(module_name: str, target_name: str):
    """
    Reads the source code of the specified logic (function, method, or class) within a given module.

    Args:
        module_name (str): The name of the module (e.g., 'agent_module').
        target_name (str): The name of the function, method, or class (e.g., 'solver', 'Agent.action_call_llm', 'Agent').

    Returns:
        code_str (str): A string representing the source code of the specified logic.
    """
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)

        # If target_name contains a dot, it's a method in a class (e.g., 'Agent.evolve')
        if '.' in target_name:
            class_name, target_name = target_name.split('.')
            target_class = getattr(module, class_name)
            target = getattr(target_class, target_name)
        else:
            # Otherwise, it's a top-level function or class
            target = getattr(module, target_name)

        # Extract the source code using inspect
        code_str = logic.get_source_code(target, target_name)

        return code_str

    except Exception as e:
        raise e


def action_adjust_logic(module_name: str, target_name: str, new_code=str, target_type: str = 'function',
                        operation: str = 'modify'):
    """
    Modify/Add/Delete the source code of the specified logic (function, method, or class) within a given module to
    improve task-solving ability or create a tool designed specifically to assist in task-solving efficiently.

    Args:
        module_name (str): The name of the module to modify (e.g., 'agent_module').
        target_name (str): The name of the function, method, or class to do operation (e.g., 'solver').
        new_code (str): The new logic as a string (including `def` for functions or `class` or classes). For delete, it can be empty string.
        target_type (str): The type of target ('function', 'class'). Default is 'function'.
        operation (str): The type of operation to perform ('modify', 'add', or 'delete'). Default is 'modify'.

    Raises:
        ValueError: Unknown operation

    Examples:
        >>> modify_logic('agent_module', 'evolve', 'def evolve(agent):\\n    print("New evolve method")', target_type='function')
        >>> modify_logic('agent_module', 'evolve', '', target_type='function', operation='delete')
    """
    if module_name == "agent_module":
        if target_name == "solver":
            if "gpt-4o" in new_code:
                raise ValueError("ONLY model **gpt-3.5-turbo** can be used in solver.")
            if "time.sleep" in new_code:
                raise ValueError("Don't use `time.sleep` in solver.")
        if target_name == "Agent.action_call_llm":
            raise ValueError("Don't modify `action_call_llm`.")
        if target_name == "Agent.action_call_json_format_llm":
            raise ValueError("Don't modify `action_call_json_format_llm`.")

    if "import logging" in new_code or "from logging" in new_code:
        raise ValueError("Don't use `logging`.")

    # Import the module dynamically
    module = importlib.import_module(module_name)
    _target_name = target_name
    print(new_code, end='\n\n')
    # Perform the operation based on type (modify, add, delete)
    if operation in ['modify', 'add']:
        # Compile the new code within the current global and a new local dict
        locals_dict = {}
        exec(compile(new_code, f"running.{module_name}.{target_name}", "exec"), globals(), locals_dict)
        if '.' in target_name:
            class_name_, target_name_ = target_name.split('.')
            if class_name_ in locals_dict:
                new_target = getattr(locals_dict[class_name_], target_name_)
                locals_dict.pop(class_name_)
            else:
                new_target = locals_dict[target_name_]
                locals_dict.pop(target_name_)
        else:
            new_target = locals_dict[target_name]
            locals_dict.pop(target_name)
        globals().update(locals_dict)

        # Apply the new definition or value to the target
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            setattr(cls, target_name, new_target)
            getattr(cls, target_name).__source__ = new_code
            # Add or update the __source__ attribute on the class level to store the full new definition
            if not hasattr(cls, '__source__'):
                cls.__source__ = {}
                for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                    cls.__source__[name] = logic.get_source_code(method, name)
            cls.__source__[target_name] = '\n'.join(['    ' + code_line for code_line in new_code.split('\n')])

        else:  # Module level attribute
            setattr(module, target_name, new_target)
            getattr(module, target_name).__source__ = new_code

    elif operation == 'delete':
        if '.' in target_name:  # Class attribute
            class_name, target_name = target_name.split('.')
            cls = getattr(module, class_name)
            delattr(cls, target_name)
            if hasattr(cls, '__source__') and target_name in cls.__source__:
                del cls.__source__[target_name]
        else:  # Module level attribute
            delattr(module, target_name)

    else:
        raise ValueError(f"Unknown operation '{operation}'. Expected 'modify', 'add', or 'delete'.")

    return f"Successfully {operation} `{module_name}.{_target_name}`."


#功能五：运行指定代码
def action_run_code(code_type: str, code: str, timeout: float = 30.0) -> str:
    """
    Execute Python or shell code and capture the output, errors, and return value.
    (Running python code can get and store objects designed specifically to assist in task-solving efficiently, such as prompts)

    Args:
        code_type (str): The type of code to execute ('python' or 'bash').
        code (str): The code to execute as a string.
        timeout (float): Maximum execution time in seconds (default: 30.0).

    Returns:
        result_str (str): A string summarizing the output, errors, and return value.
    """

    def safe_eval(expr: str, globals_dict, locals_dict):
        """Safely evaluate an expression."""
        try:
            tree = ast.parse(expr, mode='eval')
            return eval(compile(tree, '<string>', 'eval'), globals_dict, locals_dict)
        except Exception:
            return None

    if code_type.lower() == 'python':
        output = io.StringIO()
        error_output = io.StringIO()
        return_value = None

        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error_output):
            locals_dict = {}
            exec(code, globals(), locals_dict)
            globals().update(locals_dict)

            # Safely evaluate the last expression
            return_value = safe_eval(code.splitlines()[-1], globals(), locals_dict)

        result = {
            "output": output.getvalue(),
            "errors": error_output.getvalue(),
            "return_value": return_value
        }

        output.close()
        error_output.close()

    elif code_type.lower() == 'bash':
        try:
            process = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            result = {
                "output": process.stdout,
                "errors": process.stderr,
                "return_value": process.returncode
            }
        except subprocess.TimeoutExpired:
            result = {
                "output": "",
                "errors": f"Command timed out after {timeout} seconds",
                "return_value": None
            }
        except Exception as e:
            result = {
                "output": "",
                "errors": repr(e),
                "return_value": None
            }

    else:
        return "Error: Unsupported code_type. Only 'python' and 'bash' are supported."

    # Format the result
    result_str = f"Execution Summary ({code_type.capitalize()}):\n"
    if result["output"]:
        result_str += f"Output:\n{result['output']}\n"
    if result["errors"]:
        result_str += f"Errors:\n{result['errors']}\n"
    if result["return_value"] is not None:
        result_str += f"Return Value: {result['return_value']}\n"

    return result_str or "No output, errors, or return value."


#功能六：验证在任务上的准确率
def action_evaluate_on_task(task_pth, solver):
    """
    Evaluate the current solver on the goal task samples and return the evaluation feedback.

    Returns:
        feedback (str): Evaluation feedback including valid set accuracy, test set accuray, test sample inputs, model outputs and valid sample answer.
    """
    # feedback, acc = task.evaluate(solver)
    # if acc > task_mgsm.last_test_acc:
    #     logic.store_all_logic(f"../{task_mgsm.__name__}_{round(acc, 4)}")
    #     task_mgsm.last_test_acc = acc
    # return feedback
    result = []
    total_score = 0
    task = read_task(task_pth)
    for t in task:
        temp_result = solver(t)
        print("执行后的结果是",result)
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"下面是大模型针对问题{t}返回的一份结果，满分十分 你认为可以打几分,只返回分数值，结果以“分数：你认为的分数 原因：你认为的原因” 格式返回，大模型的结果是{temp_result}"
                },
            ],
        )
        print(completion.choices[0].message.content)
        # 使用正则表达式找到分数
        match = re.search(r"分数：(\d+)", completion.choices[0].message.content)
        if match:
            score = int(match.group(1))
            total_score+=score
            print("提取的分数是:", score)
        else:
            total_score += 5
            print("没有找到分数")

        result.append(f"input question is {t},output answer is {temp_result},The evaluation result is {completion.choices[0].message.content}")
    return result, total_score























