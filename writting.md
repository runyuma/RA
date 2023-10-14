## exploration RL
1. Exploration from Demonstration for Interactive Reinforcement Learning
- human demonstration to guide exploration in model free policy search exploration
2. Intrinsic Language-Guided Exploration for Complex Long-Horizon Robotic Manipulation Tasks
- use LLM as intrinsic assistant to guide exploration?
- others?
3. LaGR-SEQ: Language-Guided Reinforcement Learning with Sample-Efficient Querying
- present a frame work to prompt LLM to guide exploration for RL
- train a secondery RL agent to query LLM for exploration
# llm vlm
1. VoxPoser
## LLM for planning 
## LLM and RL
1. REWARD DESIGN WITH LANGUAGE MODELS
2. Language to Rewards for Robotic Skill Synthesis

## abstration
TBD
## Introduction
Reinforcement Learning(RL)[] is a powerful tool for learning complex decision making and control policy for Robotics[]. However, the learning process is often slow and inefficient as it requires a large amount of interaction with the environment. In the context of Robotics manipulation tasks, Deep Reinforcement Learning often have large state and action space[], which makes it hard to explore the environment efficiently.For task with long horizon, the agent often take a long time before collecting the first successful episode, a limitation that restricts practical real-world applications.\
Exploration method such as $\epsilon$-greedy[] and Boltzmann exploration[] are often inefficient for complex tasks with long horizon as they explored in a stochastic manner. In addition, these methods are often not sample efficient as they require a large amount of interaction with the environment to learn a good policy. Demonstrations from human expert can be used to guide the exploration process, as human have a large amount of prior knowledge about the environment. Subramanian et al.[EfD] proposed a method to use human demonstration to guide exploration in model free policy search exploration. However, human demonstration might be expensive to obtain and might not be available for all tasks. {Add other demonstration based exploration method}\ 
Large Language Models(LLM) such as gpt-3[] demonstrate the ability to generate human-like common sense aware reasoning.Huang et al.[] demonstrated the ability to use LLM as Zero-shot planner to decompose high-level tasks into step-by-step plans without any further training. In context of Robotics manipulation, Saycan [] extracts and leverages the knowledge within LLMs to provide task-grounding for decision making given a high level languange goal. Socratic model[] also utilize knowledge from different domains from LLM and Vision Languange model (VLM) to achieve robot perception and planing for manipulation tasks. Except for step-by-step planning, Voxposer [] extract affordance and constraints from LLM and VLM to compose a voxel value map in 3D observation for environment. Robot can utlize this value map for interaction and model-based planning.Although language model can achieve action-level planning for Robotics tasks, most prior work still rely on pretrained low level skills and manually designed motion primitives, which is often a bottleneck of the system. 

By integrating LLM and RL, this combination can alleviate the limitations of each component. LLM contributes common-sense aware reasoning to enhance RL exploration, while RL refines low-level skills through interactions with the environment. Prior work[] deployed LLM as a proxy reward signel for reinforcement learning.


LLM and RL, reward, exploration
our method LLM VLM exploration


