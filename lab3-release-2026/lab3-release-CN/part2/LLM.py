import re
from Maze import Maze
from openai import OpenAI


# TODO: Replace this with your own prompt.
your_prompt = """
你是一个吃豆人游戏智能体。你的唯一目标是在不撞墙、不撞鬼的前提下，用尽可能少的步数吃完所有豆子。

游戏规则：
1. 坐标格式为(row, col)，从0开始。
2. row减少表示向上，row增加表示向下；col减少表示向左，col增加表示向右。
3. 动作编码：
   上=0，即移动到(row-1, col)
   下=1，即移动到(row+1, col)
   左=2，即移动到(row, col-1)
   右=3，即移动到(row, col+1)
4. 墙壁不能进入。
5. 鬼魂不能进入。鬼魂不会移动，但鬼魂所在格子必须视为障碍。
6. 只能从“可用方向”中选择动作。
7. 如果选择了不可用方向、撞墙或撞鬼，游戏立即失败。
8. 豆子是目标，需要尽快吃完所有豆子。

决策方法：
1. 读取当前吃豆人位置、鬼魂位置、所有豆子位置、墙壁位置、曾经走过的位置和可用方向。
2. 对每个可用方向，计算移动后的新坐标。
3. 排除会进入墙壁、鬼魂、越界位置的动作。
4. 如果某个安全动作能直接吃到豆子，优先选择它。
5. 如果不能直接吃豆子，选择能让吃豆人到最近豆子最短的安全动作。
6. 距离按曼哈顿距离估计：abs(row1-row2)+abs(col1-col2)。
7. 如果多个动作距离相同，优先选择移动后位置不在“曾经走过的位置”里的动作。
8. 如果仍然相同，选择更靠近豆子密集区域的动作。
9. 不要反复在两个位置之间来回移动，除非其他动作都会更差或不安全。
10. 不要因为鬼魂存在而过度绕路；只要不进入鬼魂格子即可。

输出要求：
1. 必须严格按照给定格式输出。
"""

# Don't change this part.
output_format = """
输出必须是0-3的整数，上=0，下=1，左=2，右=3。
*重点*：(5,5)的上方是(4,5)，下方是(6,5)，左方是(5,4)，右方是(5,6)。
输出格式为：
“分析：XXXX。
动作：0（一个数字，不能出现其他数字）。”
"""

prompt = your_prompt + output_format


def get_game_state(maze: Maze, places: list, available: list) -> str:
    """
    Convert game state to natural language description.
    """
    description = ""
    for i in range(maze.height):
        for j in range(maze.width):
            description += f"({i},{j})="
            if maze.grid[i, j] == 0:
                description += f"空地"
            elif maze.grid[i, j] == 1:
                description += "墙壁"
            else:
                description += "豆子"
            description += ","
        description += "\n"
    places_str = ','.join(map(str, places))
    available_str = ','.join(map(str, available))
    state = f"""当前游戏状态（坐标均以0开始）：\n1、迷宫布局（0=空地,1=墙,2=豆子）：\n{description}\n2、吃豆人位置：{maze.pacman_pos[4]}\n3、鬼魂位置：{maze.pacman_pos[3]}\n4、曾经走过的位置：{places_str}\n5、可用方向：{available_str}\n"""
    return state


def get_ai_move(client: OpenAI, model_name: str, maze: Maze, file, places: list, available: list) -> int:
    """
    Get the move from the AI model.
    :param client: OpenAI client instance.
    :param model_name: Name of the AI model.
    :param maze: The maze object.
    :param file: The log file to write the output.
    :param places: The list of previous positions.
    :param available: The list of available directions.
    :return: The direction chosen by the AI.
    """
    state = get_game_state(maze, places, available)

    file.write("________________________________________________________\n")
    file.write(f"message:\n{state}\n")
    print("________________________________________________________")
    print(f"message:\n{state}")

    print("Waiting for AI response...")
    all_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": state
            }
        ],
        stream=False,
        temperature=.0
    )
    info = all_response.choices[0].message.content

    file.write(f"AI response:\n{info}\n")
    print(f"AI response:\n{info}")
    numbers = re.findall(r'\d+', info)
    choice = numbers[-1]
    return int(choice), info
