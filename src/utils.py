import yaml
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    """
    加载并解析YAML配置文件。

    Args:
        path (str): YAML文件的路径。

    Returns:
        Dict[str, Any]: 包含配置信息的字典。
    
    Raises:
        FileNotFoundError: 如果指定的路径下文件不存在。
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 '{path}' 未找到。")
        raise
    except yaml.YAMLError as e:
        print(f"错误: 解析YAML文件 '{path}' 时出错: {e}")
        raise

if __name__ == '__main__':
    # 创建一个虚拟的配置文件用于演示
    dummy_config_path = 'configs/dummy_config.yaml'
    dummy_config_content = """
# 模型相关配置
model:
  name: "GRUModel"
  embed_dim: 128
  hidden_dim: 256
  output_dim: 2
  n_layers: 2
  bidirectional: true
  dropout: 0.5

# 训练相关配置
training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 10
  device: "cuda"

# 数据相关配置
data:
  path: "data/ChnSentiCorp_htl_all.csv"
  min_freq: 5
"""
    with open(dummy_config_path, 'w', encoding='utf-8') as f:
        f.write(dummy_config_content)

    print(f"创建虚拟配置文件于: {dummy_config_path}")
    
    # 加载并打印配置
    try:
        config = load_config(dummy_config_path)
        print("\n加载的配置内容:")
        import json
        print(json.dumps(config, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"加载配置失败: {e}") 