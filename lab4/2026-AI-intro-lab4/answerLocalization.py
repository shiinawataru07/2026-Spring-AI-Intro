from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 0.75
MAX_ERROR = 50000
K_WEIGHT = 0.95
SIGMA_XY = 0.10
SIGMA_THETA = 0.08

### 可以在这里写下一些你需要的变量和函数 ###

def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    ### 你的代码 ###
    def posValid(x, y):
        dx = np.abs(walls[:, 0] - x)
        dy = np.abs(walls[:, 1] - y)
        return not np.any((dx < COLLISION_DISTANCE) & (dy < COLLISION_DISTANCE))

    x_min = np.min(walls[:, 0]) + 0.5
    x_max = np.max(walls[:, 0]) - 0.5
    y_min = np.min(walls[:, 1]) + 0.5
    y_max = np.max(walls[:, 1]) - 0.5

    while len(all_particles) < N:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        theta = np.random.uniform(-np.pi, np.pi)
        if posValid(x, y):
            all_particles.append(Particle(x, y, theta, 1.0 / N))
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    ### 你的代码 ###
    distance = np.sqrt(np.sum((estimated - gt) ** 2))
    weight = np.exp(-K_WEIGHT * distance)
    ### 你的代码 ###
    return float(weight)


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    ### 你的代码 ###
    n = len(particles)
    def posValid(x, y):
        dx = np.abs(walls[:, 0] - x)
        dy = np.abs(walls[:, 1] - y)
        return not np.any((dx < COLLISION_DISTANCE) & (dy < COLLISION_DISTANCE))
    weights = np.array([p.weight for p in particles], dtype = float)
    weights /= np.sum(weights)
    randomIdx = np.random.choice(n, size = n, p = weights)
    top_weight_sum = np.sum(weights[:max(1, n // 10)])
    if top_weight_sum > 0.75:
        local_ratio = 0.995
    elif top_weight_sum > 0.60:
        local_ratio = 0.98
    elif top_weight_sum > 0.45:
        local_ratio = 0.95
    elif top_weight_sum > 0.30:
        local_ratio = 0.90
    else:
        local_ratio = 0.85
    for idx in randomIdx[:int(n * local_ratio)]:
        p = particles[idx]
        new_x = p.position[0] + np.random.normal(0, SIGMA_XY)
        new_y = p.position[1] + np.random.normal(0, SIGMA_XY)
        new_theta = p.theta + np.random.normal(0, SIGMA_THETA)
        cnt = 0
        if posValid(new_x, new_y):
            resampled_particles.append(Particle(new_x, new_y, new_theta, 1.0 / n))
        else:
            while cnt < 10:
                new_x = p.position[0] + np.random.normal(0, SIGMA_XY)
                new_y = p.position[1] + np.random.normal(0, SIGMA_XY)
                new_theta = p.theta + np.random.normal(0, SIGMA_THETA)
                if posValid(new_x, new_y):
                    resampled_particles.append(Particle(new_x, new_y, new_theta, 1.0 / n))
                    break
                cnt += 1
    x_min = np.min(walls[:, 0]) + 0.5
    x_max = np.max(walls[:, 0]) - 0.5
    y_min = np.min(walls[:, 1]) + 0.5
    y_max = np.max(walls[:, 1]) - 0.5
    while len(resampled_particles) < n:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        theta = np.random.uniform(-np.pi, np.pi)
        if posValid(x, y):
            resampled_particles.append(Particle(x, y, theta, 1.0 / n))
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta = (p.theta + dtheta) % (2 * np.pi)
    p.position[0] += traveled_distance * np.cos(p.theta)
    p.position[1] += traveled_distance * np.sin(p.theta)
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    ### 你的代码 ###
    best = particles[0]
    theta = np.arctan2(np.sin(best.theta), np.cos(best.theta))
    final_result = Particle(best.position[0], best.position[1], theta, 1.0)
    ### 你的代码 ###
    return final_result