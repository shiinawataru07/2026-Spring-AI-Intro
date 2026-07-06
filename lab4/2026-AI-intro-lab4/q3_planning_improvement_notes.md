# Q3 运动规划提分建议

## 当前现象

你现在的双向 RRT 在多数 case 上可以工作，但 case 4 和 case 5 会出现“卡入墙角后不再移动”的问题。根据代码阅读和对 `test_idx=10`、`test_idx=11` 的复现，主要问题不是 RRT 完全找不到路径，而是生成的路径点在仿真控制里不够好跟踪。

典型失败形态是：

- 某一段路径的下一个 target 离当前 Pacman 位置较远。
- 这段直线在规划层面可能是可行或接近可行的，但 PD 控制会把 Pacman 斜向推到墙角。
- Pacman 速度变成 0 后，`get_target` 仍然返回同一个墙后目标点。
- 控制器持续往墙里推，于是一直卡住，直到超过 `max_step`。

所以优先提分方向不是继续单纯增加 RRT 迭代次数，而是让 planner 输出“仿真中可执行”的路径。

## 优先级 1：改 `get_target` 的目标选择策略

当前逻辑里有一行很激进：

```python
if self.map.checkline(current_position.tolist(), self.goal.tolist())[0] == False:
    return self.goal
```

这会导致只要当前位置到最终食物没有射线碰撞，就直接冲终点。问题是 Box2D 里的 Pacman 有半径、有速度、有惯性，直线可见不代表 PD 一定能稳定通过窄口或墙角。

建议改成：

1. 只有当距离终点较近，例如 `< 2.0` 或 `< 2.5`，并且直线可见时，才直接返回 `goal`。
2. 否则仍然沿当前路径点走。
3. 路径点不要一次跳太远，可以用 lookahead，但必须限制距离和可见性。

可以尝试的参数：

```python
TARGET_THREHOLD = 0.30  # 或 0.35
LOOKAHEAD_DISTANCE = 2.0
```

思路是：Pacman 接近当前路径点后再切下一个点；如果后面 1 到 3 个路径点在短距离内且直线可见，可以提前看过去，但不要直接看完整个终点。

## 优先级 2：加“卡住检测 + 重规划”

当前代码没有处理 Pacman 实际走偏的情况。README 也提示了：仿真中的 Pacman 不一定能准确到达指定位置，必要时需要重新规划。

建议在 `RRT.__init__` 里加：

```python
self.stuck_counter = 0
```

在 `find_path` 里重置：

```python
self.stuck_counter = 0
```

在 `get_target` 里判断：

```python
if np.linalg.norm(current_velocity) < 0.03 and np.linalg.norm(current_position - self.goal) > 0.8:
    self.stuck_counter += 1
else:
    self.stuck_counter = 0

if self.stuck_counter > 90:
    self.path = self.build_tree(current_position, self.goal)
    self.path_idx = 0
    self.stuck_counter = 0
```

这样 Pacman 如果被墙角卡住 1.5 秒左右，就会从当前位置重新规划，而不是一直推墙。

## 优先级 3：把 RRT 路径变成更容易跟踪的路径

你现在的 RRT 路径点有两个问题：

- 有些点太贴墙或转角太急。
- 路径拼接后可能出现较长斜线段，PD 控制容易在拐角处切弯撞墙。

有两个改法，建议先选简单稳定的。

### 方案 A：保留 RRT，但做路径后处理

在 `build_tree` 返回前，对路径做两步处理：

1. 路径压缩：如果 `path[i]` 到 `path[j]` 可见，就跳过中间点。
2. 重新插值：压缩后的每段不要超过 `0.5` 或 `0.75`，把长线段拆成小 target。

注意：压缩不要太激进。对 case 4/5，过度压缩反而容易在墙角切弯。

比较稳的做法是：

- 只允许压缩到短距离可见点，例如每次最多看 3 到 5 个点。
- 每段插值长度控制在 `0.5` 左右。

### 方案 B：用网格 A* 替代 RRT 输出

因为地图本身是格子地图，食物和墙也都在格点附近，Q3 其实可以用 A* 直接找格心路径。优点是确定、稳定、不贴随机斜线。

基本做法：

1. 把墙转成 `wall_set = {(x, y), ...}`。
2. 起点和终点 round 到最近格点。
3. 用四邻域 A* 搜索，不走墙格。
4. 返回格心路径点。
5. 在 `get_target` 里做短距离 lookahead。

这个方案对 case 4/5 通常比 RRT 更稳，缺点是路径可能更长，步数不一定最优。但你现在 case 4/5 是失败低分，先保证全吃完，分数会明显上升。

## 优先级 4：修正双向 RRT 的连接效率

当前 `meet_graph` 会遍历另一棵树里所有节点，并且只要任意节点可见就连接：

```python
for i, node in enumerate(graph):
    if self.map.checkline(point.tolist(), node.pos.tolist())[0] == False:
        return i
```

这可能连接到一个很远的节点，导致路径中出现大跨度段。建议改成：

1. 先找另一棵树最近节点。
2. 只有最近节点距离小于某个阈值，例如 `1.0` 或 `1.5`，并且直线可见，才连接。

这样路径会更局部、更平滑，也更容易被 PD 跟踪。

## 我建议的修改顺序

1. 先改 `get_target`：限制直冲终点，加短距离 lookahead。
2. 再加卡住检测和重规划。
3. 如果 case 4/5 仍失败，再给 RRT 路径加后处理。
4. 如果还不稳，直接把 `build_tree` 的主路径生成换成 A*，保留 `RRT` 类接口不变。

## 验证方式

每次改完建议先跑：

```powershell
python taskPlanning.py --no_render --test_idx 10 --max_step 20000
python taskPlanning.py --no_render --test_idx 11 --max_step 20000
python autograder.py --q q3
```

目标顺序：

1. case 4/5 先从失败变成成功。
2. 再看总步数是否能降到 `18000` 以下。
3. 最后再优化路径长度，争取接近 `10000` 到 `12000`。

## 结论

最值得先改的是 `get_target`，不是 `MAX_ITER`。case 4/5 的低分核心是“规划路径能找到，但仿真控制跟踪失败”。只要让目标点更短、更居中，并且在卡住时能重规划，分数大概率会比现在的 86 明显提升。
