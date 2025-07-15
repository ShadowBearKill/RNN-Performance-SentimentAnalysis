# RNN 家族模型性能对比研究项目 - 系统架构设计

本文档详细描述了用于循环神经网络（RNN）家族模型在情感分析任务上性能对比的系统架构设计。

## 1. 项目概述

本项目旨在系统性地比较基础RNN、LSTM、GRU及其多种变体（多层、双向、自注意力机制）在中文情感分析任务上的性能表现。为确保研究的科学性和公平性，我们设计了一个统一的实验框架，涵盖数据处理、模型构建、训练、评估和结果分析的全过程。

## 2. 系统整体架构

### 2.1. 架构图

系统采用模块化设计，分为数据层、模型层、实验层和评估分析层，各层职责分明，相互协作。

```mermaid
graph TD
    subgraph "数据层 (Data Layer)"
        A[原始数据集<br/>ChnSentiCorp_htl_all] --> B{数据处理流水线};
        B --> C[文本清洗 & 分词];
        C --> D[构建词典 & 向量化];
        D --> E[序列填充 & 批处理];
        E --> F[PyTorch DataLoader];
    end

    subgraph "模型层 (Model Zoo)"
        G[模型基类 BaseRNN];
        H[基础 RNN] --继承--> G;
        I[LSTM] --继承--> G;
        J[GRU] --继承--> G;
        K["双向 & 多层装饰器"] --包装--> J;
        L["自注意力模块"] --组合--> J;
    end

    subgraph "实验层 (Experiment Layer)"
        M[实验配置 (YAML)];
        N[实验执行器 (Trainer)];
        O[训练/验证/测试循环];
        M --传入配置--> N;
        F --提供数据--> N;
        G --实例化--> N;
        N --> O;
    end

    subgraph "评估与分析层 (Evaluation & Analysis)"
        P[评估模块];
        Q[结果记录器<br/>(CSV/MLflow)];
        R[结果可视化<br/>(Matplotlib/Seaborn)];
        S[性能指标计算<br/>准确率, F1, 推理效率];
        T[最终对比表];
        O --产出模型/日志--> P;
        P --> S;
        S --> Q;
        Q --> R;
        R --> T;
    end
```

### 2.2. 核心设计思想

-   **模块化 (Modularity)**：系统被划分为独立的模块，每个模块负责单一的功能。这种设计降低了模块间的耦合度，便于独立开发、测试和维护。
-   **可扩展性 (Extensibility)**：模型层采用基类和装饰器/组合模式，未来添加新的RNN变体（如Attention-Bi-GRU-2L）只需编写少量代码，而无需修改核心训练和评估逻辑。
-   **可复现性 (Reproducibility)**：通过统一的实验配置、固定的随机种子和详细的日志记录，确保任何一次实验都可以被精确复现。

## 3. 模块详细设计

### 3.1. 数据处理流水线 (Data Pipeline)

此模块负责将原始文本数据转换为模型可用的张量。

-   **输入**：原始 `ChnSentiCorp_htl_all` 数据集。
-   **处理流程**：
    1.  **加载数据**：从文件中读取评论文本和对应的情感标签（正面/负面）。
    2.  **文本清洗**：去除HTML标签、特殊字符和不必要的空格。
    3.  **分词**：使用 `jieba` 等中文分词工具对文本进行分词。
    4.  **构建词典**：遍历所有训练文本，构建一个从词到索引的映射词典。可以设定最小词频以过滤低频词。
    5.  **向量化**：将每个分词后的句子转换为对应的索引序列。
    6.  **序列填充 (Padding)**：由于句子长度不一，将所有序列填充到统一的长度（或批处理中的最大长度）。
    7.  **数据划分**：将数据集划分为训练集、验证集和测试集。
    8.  **数据加载器**：使用 PyTorch `DataLoader` 创建数据迭代器，用于模型训练时的批处理和数据混洗。
-   **输出**：可直接输入模型的批处理数据 `(batch_sequences, batch_labels, batch_lengths)`。

### 3.2. 可扩展模型架构 (Model Zoo)

这是项目的核心，包含所有待测试的RNN模型。

-   **`BaseModel` (模型基类)**：定义一个所有模型都必须遵循的接口，例如 `forward()` 方法。它将包含一个嵌入层（`nn.Embedding`）和一个最终的分类器（`nn.Linear`）。
-   **核心RNN模块**：
    -   `RNN`, `LSTM`, `GRU` 作为基础实现。
-   **模型变体实现**：
    1.  **基础模型** (`BasicRNN`, `LSTM`, `GRU`)：直接调用PyTorch相应的基础模块。
    2.  **多层模型** (`GRU-2L`)：在实例化 `nn.GRU` 时，设置 `num_layers=2`。
    3.  **双向模型** (`BiGRU`)：在实例化 `nn.GRU` 时，设置 `bidirectional=True`。分类器的输入维度需要加倍。
    4.  **双向多层模型** (`BiGRU-2L`)：同时设置 `num_layers=2` 和 `bidirectional=True`。
    5.  **自注意力模型** (`SelfAttentionGRU`)：在GRU层后添加一个自注意力（Self-Attention）层。该注意力层将GRU的输出作为Query, Key, Value，计算加权和，然后将结果送入分类器。
-   **配置驱动**：模型的具体参数（如隐藏层大小、层数、是否双向）将通过配置文件传入，使得模型实例化过程高度灵活。

### 3.3. 实验执行器 (Experiment Runner)

负责管理整个实验流程，从模型训练到评估。

-   **输入**：实验配置文件（YAML格式），包含模型类型、超参数（学习率、批大小、epoch数）、随机种子等。
-   **核心功能**：
    -   **环境设置**：根据配置设置随机种子，确保实验可复现。
    -   **组件初始化**：根据配置初始化数据加载器、模型、优化器（Adam/SGD）和损失函数（CrossEntropyLoss）。
    -   **训练循环**：执行指定的epoch数量。在每个epoch中：
        -   在训练集上迭代，计算损失，反向传播，更新模型权重。
        -   在验证集上评估模型性能，用于早停（Early Stopping）和模型选择。
    -   **日志记录**：记录每个epoch的训练损失、验证损失和评估指标。推荐使用`MLflow`或`Weights & Biases`进行自动化跟踪。
    -   **模型保存**：保存验证集上性能最佳的模型权重。
-   **输出**：训练好的模型文件和详细的实验日志。

### 3.4. 评估与结果分析 (Evaluation & Analysis)

在模型训练完成后，此模块负责进行最终的性能评估和结果汇总。

-   **测试**：加载性能最佳的模型，在独立的测试集上进行最终评估。
-   **性能指标**：
    -   **效果指标**：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1分数。
    -   **效率指标**：计算模型在测试集上的总推理时间和平均单样本推理时间。
-   **结果存储**：将每次实验的配置、最终评估指标和模型路径记录到一个中心化的CSV文件或数据库中。
-   **结果分析**：从记录文件中读取所有模型的实验结果，进行汇总和分析。

## 4. 目录结构规划

一个清晰的目录结构对于项目维护至关重要。

```
RNN_COMPARISON/
│
├── configs/                  # 存放所有实验的配置文件 (YAML)
│   ├── base_rnn.yaml
│   ├── gru.yaml
│   └── lstm.yaml
│
├── data/                     # 存放原始数据集和处理后的数据
│   └── ChnSentiCorp_htl_all.csv
│
├── notebooks/                # Jupyter Notebooks，用于探索性分析和结果可视化
│   └── visualize_results.ipynb
│
├── src/                      # 项目源代码
│   ├── __init__.py
│   ├── data_loader.py        # 数据处理和加载模块
│   ├── models/               # 模型定义
│   │   ├── __init__.py
│   │   ├── base_model.py     # 模型基类
│   │   ├── rnn.py
│   │   ├── lstm.py
│   │   └── gru.py            # GRU, Self-Attention GRU等
│   ├── trainer.py            # 实验执行器
│   ├── evaluate.py           # 评估模块
│   └── utils.py              # 辅助函数（如日志、随机种子设置）
│
├── results/                  # 存放实验结果
│   ├── logs/                 # 训练日志
│   ├── saved_models/         # 保存的模型权重
│   └── summary.csv           # 汇总的实验结果
│
├── main.py                   # 项目主入口，解析参数，启动实验
├── requirements.txt          # 项目依赖
└── SYSTEM_ARCHITECTURE.md    # 本文档
```

## 5. 技术栈与依赖管理

-   **编程语言**: Python 3.8+
-   **核心框架**: PyTorch 1.10+
-   **中文分词**: Jieba
-   **数据处理**: Pandas, NumPy
-   **可视化**: Matplotlib, Seaborn
-   **实验跟踪 (推荐)**: MLflow
-   **依赖管理**: 使用 `pip` 和 `requirements.txt`。

`requirements.txt` 文件内容示例：
```
torch
pandas
numpy
jieba
matplotlib
seaborn
pyyaml
mlflow
scikit-learn
```

## 6. 实验流程设计

1.  **准备数据**：将 `ChnSentiCorp_htl_all` 数据集放置在 `data/` 目录下。
2.  **创建配置**：在 `configs/` 目录下为每个待测试的模型创建一个YAML配置文件，定义模型参数和训练超参数。
3.  **执行实验**：运行主脚本 `main.py`，通过命令行参数指定要使用的配置文件。
    ```bash
    python main.py --config configs/gru.yaml
    ```
4.  **自动化执行**：可以编写一个简单的shell脚本来依次执行所有模型的配置文件，实现全自动化实验。
    ```bash
    for config in configs/*.yaml; do
        python main.py --config "$config"
    done
    ```
5.  **分析结果**：实验完成后，所有结果都记录在 `results/summary.csv` 中。运行 `notebooks/visualize_results.ipynb` 来加载、分析和可视化结果。

## 7. 结果可视化方案

最终的实验结果将通过一个清晰的二维表格进行展示，并辅以图表。

-   **二维表格**：
    -   **行**：代表9种不同的RNN模型。
    -   **列**：代表不同的评估指标（准确率, Precision, Recall, F1, 平均推理时间）。
    -   使用 `pandas` DataFrame 创建此表格，并可以导出为Markdown或图片格式。

| 模型 | 准确率 (%) | F1 分数 | 平均推理时间 (ms/sample) |
| :--- | :--- | :--- | :--- |
| 基础 RNN | | | |
| LSTM | | | |
| GRU | | | |
| 2-layer GRU | | | |
| ... | | | |
| 双向Self-attention GRU | | | |

-   **条形图**：使用 `matplotlib` 或 `seaborn` 绘制条形图，直观比较不同模型在关键指标（如准确率和推理时间）上的表现。

## 8. 部署和使用指南

1.  **克隆仓库**：`git clone <repository_url>`
2.  **创建虚拟环境**：
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Windows: venv\Scripts\activate
    ```
3.  **安装依赖**：`pip install -r requirements.txt`
4.  **准备数据**：将数据集放入 `data/` 文件夹。
5.  **运行单个实验**：
    ```bash
    python main.py --config configs/your_model_config.yaml
    ```
6.  **查看结果**：检查 `results/` 目录下的输出和 `results/summary.csv` 中的汇总数据。

该架构设计为您的研究项目提供了一个坚实的基础，确保了流程的标准化和结果的可靠性。 