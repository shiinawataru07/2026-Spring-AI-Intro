# 人工智能引论课程实验

这是人工智能引论课程的实验代码归档，按实验编号组织。仓库内容包含课程提供的 Pacman 框架、自动评测器、数据文件、训练得到的模型，以及个人完成的算法实现。

主要覆盖内容包括 Python 与 NumPy 基础、经典搜索、多智能体博弈、机器学习、自动微分、神经网络、自然语言处理、大模型提示词控制、机器人定位、运动控制与路径规划。

<<<<<<< HEAD
=======
期望能为其他学习者带来便利。

>>>>>>> 66034226dc7c2bffe7db221cbb3b9f18e4e8ecc2
> 说明：本仓库适合作为个人学习记录和本地 Git 管理。部分实验基于 UC Berkeley CS188、Harvard CS50 AI 或课程组改编框架，请遵守课程与原项目的使用约束。

## 目录结构

```text
.
├── AI-lab0-final/                 # Lab 0：Python、NumPy、类与简单 Pacman 地图练习
├── numpy_matplotlib/              # NumPy / Matplotlib / SVD 可视化练习
├── lab1-part1-final/              # Lab 1 Part 1：搜索算法
│   └── search/
├── lab1-part2/                    # Lab 1 Part 2：多智能体 Pacman
├── AIIntroLab2/                   # Lab 2：机器学习、自动微分、MNIST 视觉模型
├── lab3-release-2026/
│   └── lab3-release-CN/           # Lab 3：NLP、QA 与 LLM 控制 Pacman
└── lab4/
    └── 2026-AI-intro-lab4/        # Lab 4：定位、PD 控制与路径规划
```

## 实验概览

| 实验 | 主题 | 主要实现/入口 |
| --- | --- | --- |
| Lab 0 | Python 基础、NumPy、地图解析 | `AI-lab0-final/exercise_0.py` |
| NumPy/Matplotlib | SVD 截断、矩阵重构、图像显示 | `numpy_matplotlib/numpy_and_matplotlib.py` |
| Lab 1 Part 1 | DFS、BFS、UCS、A*、Corners 启发式 | `lab1-part1-final/search/search.py`, `searchAgents.py` |
| Lab 1 Part 2 | Minimax、Alpha-Beta、MCTS | `lab1-part2/multiAgents.py` |
| Lab 2 | Logistic Regression、Decision Tree、Random Forest、Softmax、MLP、CNN | `AIIntroLab2/answer*.py`, `autograd/`, `MnistModel.py`, `YourTraining.py` |
| Lab 3 Part 1 | Naive Bayes 情感分类、Attention 分类、TF-IDF QA | `lab3-release-2026/lab3-release-CN/part1/FruitModel.py` |
| Lab 3 Part 2 | LLM 提示词驱动 Pacman 决策 | `lab3-release-2026/lab3-release-CN/part2/LLM.py`, `main.py` |
| Lab 4 | 粒子滤波定位、PD 控制、RRT 路径规划 | `lab4/2026-AI-intro-lab4/answerLocalization.py`, `answerPDControl.py`, `answerPlanning.py` |

## 环境建议

大多数实验使用 Python 3.10 或 3.11。建议为课程实验创建单独环境：

```powershell
conda create -n ai-intro python=3.10
conda activate ai-intro
pip install numpy scipy matplotlib pillow nltk
```

部分实验有额外依赖：

```powershell
# Lab 3 Part 2
pip install pygame openai pycryptodome

# Lab 4
pip install panda3d scipy
```

Lab 4 的 `Box2D` 在 Windows 下已随实验目录提供多个 Python 版本的预编译文件；如果在 Linux/macOS 上运行，需要按该实验子目录 README 的说明安装 `box2d-py`。

## 常用运行方式

每个实验通常需要进入对应目录运行，避免相对路径找不到数据或模型。

### Lab 1 Part 1：搜索

```powershell
cd lab1-part1-final\search
python autograder.py
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs
python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```

### Lab 1 Part 2：多智能体

```powershell
cd lab1-part2
python autograder.py
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=2
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
python pacman.py -p MCTSAgent -l testClassic
```

### Lab 2：机器学习与视觉模型

```powershell
cd AIIntroLab2
python autograder.py
python modelLogisticRegression.py
python modelTree.py
python modelRandomForest.py
python modelSoftmaxRegression.py
python modelMultiLayerPerceptron.py
python YourTraining.py
```

可视化 Pacman 表现示例：

```powershell
python pacman.py -p ReflexAgent -a model="MLP" -l accuracy --maxstep 40
python pacman.py -p ReflexAgent -a model="Your" -l smallPKU
```

### Lab 3：自然语言处理与大模型控制

```powershell
cd lab3-release-2026\lab3-release-CN\part1
python autograder.py
python FruitModel.py
python pacman.py -p ReflexAgent -a model="Naive" -l fruit --maxstep 50
python pacman.py -p ReflexAgent -a model="QA" -l fruitqa --maxstep 50
python qa.py
```

```powershell
cd lab3-release-2026\lab3-release-CN\part2
python main.py
python autograder.py
```

Lab 3 Part 2 需要在 `main.py` 中配置可用的大模型 API Key。

### Lab 4：机器人定位、控制与规划

```powershell
cd lab4\2026-AI-intro-lab4
python autograder.py
python autograder.py --q q1
python autograder.py --q q2
python autograder.py --q q3
```

单项可视化示例：

```powershell
python taskLocalization.py --test_idx 0
python taskPDControl.py --test_idx 0
python taskPlanning.py --test_idx 6
```

## 实现要点

- 搜索：实现了 DFS、BFS、UCS、A*，并在四角问题中使用基于曼哈顿距离的启发式估计。
- 多智能体：实现了 Minimax、Alpha-Beta 剪枝和 MCTS 框架中的选择、扩展、模拟与回传逻辑。
- 机器学习：实现了逻辑回归、决策树、随机森林、Softmax 回归、多层感知机和自定义 CNN 视觉模型。
- 自动微分：补全了计算图前向传播、反向传播、参数更新，以及多类神经网络节点。
- NLP：实现了朴素贝叶斯情感分类、词向量嵌入、Attention 分类模型和基于 TF-IDF 的问答检索。
- LLM 控制：编写了面向 Pacman 迷宫任务的系统提示词与输出解析流程。
- 机器人：实现了粒子滤波定位、PD 控制器和双向 RRT 路径规划，并加入卡住检测与重规划机制。

<<<<<<< HEAD
## Git 管理建议

当前目录可作为一个整体仓库初始化：

```powershell
git init
git add README.md
git add AI-lab0-final numpy_matplotlib lab1-part1-final lab1-part2 AIIntroLab2 lab3-release-2026 lab4
git commit -m "Initial course lab archive"
```

建议后续补充 `.gitignore`，排除 Python 缓存、临时日志和不需要版本化的中间产物：

```gitignore
__pycache__/
*.pyc
*.log
.DS_Store
```

如果希望仓库更轻，可以进一步评估是否保留训练模型、压缩包、数据集和可视化资源；如果目标是完整课程归档，则保留它们更方便复现实验结果。

=======
>>>>>>> 66034226dc7c2bffe7db221cbb3b9f18e4e8ecc2
## 备注

子目录中保留了课程原始 README、PDF、自动评测器和测试数据。根目录 README 只提供总览和常用入口；更细的题目要求、评分方式和环境差异请参考各实验目录中的说明文件。
