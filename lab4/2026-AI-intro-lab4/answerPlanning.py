import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 0.5
TARGET_THREHOLD = 1.0
MAX_ITER = 10000
LOOKHEAD_DISTANCE = 10.0
SPEED_THRESHOLD = 0.03

# LOOKHEAD_DISTANCE = 4.0, TARGET_THREHOLD = 0.25, SUCCESS_RATE = 89
# LOOKHEAD_DISTANCE = 2.5, TARGET_THREHOLD = 0.30, SUCCESS_RATE = 90
# LOOKHEAD_DISTANCE = 2.0, TARGET_THREHOLD = 0.30, SUCCESS_RATE = 90
# LOOKHEAD_DISTANCE = 10.0, TARGET_THREHOLD = 0.25, SUCCESS_RATE = 92
# LOOKHEAD_DISTANCE = 10.0, TARGET_THREHOLD = 0.45, SUCCESS_RATE = 96
# LOOKHEAD_DISTANCE = 10.0, TARGET_THREHOLD = 0.9, SUCCESS_RATE = 99
# LOOKHEAD_DISTANCE = 10.0, TARGET_THREHOLD = 2.0, SUCCESS_RATE = 99
# LOOKHEAD_DISTANCE = 10.0, TARGET_THREHOLD = 1.1, SUCCESS_RATE = 99
# LOOKHEAD_DISTANCE = 10.0, TARGET_THREHOLD = 1.0, SUCCESS_RATE = 100
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.x_min = np.min(walls[:, 0]) + 0.5
        self.x_max = np.max(walls[:, 0]) - 0.5
        self.y_min = np.min(walls[:, 1]) + 0.5
        self.y_max = np.max(walls[:, 1]) - 0.5
        self.path_idx = 0
        self.path = None
        self.goal = None
        self.stuck_counter = 0
        ### 你的代码 ###
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        self.goal = next_food
        self.path_idx = 0
        self.stuck_counter = 0
        ### 你的代码 ###
        # 如有必要，此行可删除
        self.path = self.build_tree(current_position, next_food)
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        if self.goal is None or self.path is None or len(self.path) == 0:
            return current_position
        
        if np.linalg.norm(current_velocity) < SPEED_THRESHOLD and np.linalg.norm(current_position - self.goal) > 0.8:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 80:
            self.find_path(current_position, self.goal)
            
        distance_to_goal = np.linalg.norm(current_position - self.goal)
        if self.map.checkline(current_position.tolist(), self.goal.tolist())[0] == False and distance_to_goal < LOOKHEAD_DISTANCE:
            return self.goal
        while self.path_idx < len(self.path) - 1:
            if np.linalg.norm(current_position - self.path[self.path_idx]) < TARGET_THREHOLD:
                self.path_idx += 1
            else:
                break
        target_pose = self.path[self.path_idx]
        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        tree_path = []
        goal_path = []
        treegraph: List[TreeNode] = []
        goalgraph: List[TreeNode] = []
        treegraph.append(TreeNode(-1, start[0], start[1]))
        goalgraph.append(TreeNode(-1, goal[0], goal[1]))
        ### 你的代码 ###
        if self.map.checkline(start.tolist(), goal.tolist())[0] == False:
            path.append(goal)
            return path
        
        for _ in range(MAX_ITER):
            rand_point = np.array([np.random.uniform(self.x_min, self.x_max), np.random.uniform(self.y_min, self.y_max)])
            tree_nearest_idx, _ = self.find_nearest_point(rand_point, treegraph)
            tree_nearest_node = treegraph[tree_nearest_idx]
            is_empty, new_point = self.connect_a_to_b(tree_nearest_node.pos, rand_point)
            if is_empty:
                treegraph.append(TreeNode(tree_nearest_idx, new_point[0], new_point[1]))
                if self.meet_graph(new_point, goalgraph) != -1:
                    goal_idx = self.meet_graph(new_point, goalgraph)
                    goal_path.append(goalgraph[goal_idx].pos)
                    while goal_idx != -1:
                        goal_idx = goalgraph[goal_idx].parent_idx
                        if goal_idx != -1:
                            goal_path.append(goalgraph[goal_idx].pos)
                    tree_path.append(new_point)
                    tree_idx = tree_nearest_idx
                    tree_path.append(treegraph[tree_idx].pos)
                    while tree_idx != -1:
                        tree_idx = treegraph[tree_idx].parent_idx
                        if tree_idx != -1:
                            tree_path.append(treegraph[tree_idx].pos)
                    path = tree_path[::-1] + goal_path
                    return path[1:]
                
            tree_nearest_idx, _ = self.find_nearest_point(rand_point, goalgraph)
            tree_nearest_node = goalgraph[tree_nearest_idx]
            is_empty, new_point = self.connect_a_to_b(tree_nearest_node.pos, rand_point)
            if is_empty:
                goalgraph.append(TreeNode(tree_nearest_idx, new_point[0], new_point[1]))
                if self.meet_graph(new_point, treegraph) != -1:
                    tree_idx = self.meet_graph(new_point, treegraph)
                    tree_path.append(treegraph[tree_idx].pos)
                    while tree_idx != -1:
                        tree_idx = treegraph[tree_idx].parent_idx
                        if tree_idx != -1:
                            tree_path.append(treegraph[tree_idx].pos)
                    goal_path.append(new_point)
                    goal_idx = tree_nearest_idx
                    goal_path.append(goalgraph[goal_idx].pos)
                    while goal_idx != -1:
                        goal_idx = goalgraph[goal_idx].parent_idx
                        if goal_idx != -1:
                            goal_path.append(goalgraph[goal_idx].pos)
                    path = tree_path[::-1] + goal_path
                    return path[1:]
        
           
        ### 你的代码 ###
        return path
    
    def meet_graph(self, point, graph):
        for i, node in enumerate(graph):
            if self.map.checkline(point.tolist(), node.pos.tolist())[0] == False:
                return i
        return -1

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        for i, node in enumerate(graph):
            distance = np.linalg.norm(node.pos - point)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_idx = i
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        dir = point_b - point_a
        if np.linalg.norm(dir) < 1e-6:
            return False, point_a
        dir = dir / np.linalg.norm(dir)
        newpoint = point_a + dir * STEP_DISTANCE
        is_empty = self.map.checkline(point_a.tolist(), newpoint.tolist())[0] == False and self.map.checkoccupy(newpoint.tolist()) == False
        ### 你的代码 ###
        return is_empty, newpoint
