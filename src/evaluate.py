import torch
import pandas as pd
import time
import argparse
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import numpy as np

from src.utils import load_config
from src.data_loader import (
    load_and_preprocess_data, 
    Vocabulary, 
    vectorize_data,
    SentimentDataset,
    collate_fn
)
from src.models import GRUModel # 同样需要动态加载
from src.trainer import Trainer

def evaluate_model(config_path: str):
    """
    加载训练好的模型，并在测试集上进行最终评估。

    Args:
        config_path (str): 实验的配置文件路径。
    """
    # 1. 加载配置
    config = load_config(config_path)
    model_name = config['model']['name']
    model_save_path = f"results/saved_models/{model_name}_best.pt"
    
    # 2. 加载数据和词典
    # 注意：这里我们使用与训练时完全相同的数据处理流程
    full_data = load_and_preprocess_data(config['data']['path'])
    vocab = Vocabulary(min_freq=config['data']['min_freq'])
    vocab.build_vocab([tokens for tokens, _ in full_data])
    vectorized_data = vectorize_data(full_data, vocab)
    
    # 在真实场景中，你应该有独立的 train/valid/test 划分
    # 这里我们为了简化流程，暂时使用全部数据作为测试集
    # TODO: 在main.py中实现 train/valid/test 的分割
    test_dataset = SentimentDataset(vectorized_data)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # 3. 初始化一个临时的Trainer来使用其评估功能
    # 我们不需要它的训练功能，只需要它的模型和评估方法
    # 为了初始化Trainer，需要提供模拟的train/valid loader
    trainer = Trainer(
        config=config, vocab=vocab,
        train_loader=test_loader, # 模拟
        valid_loader=test_loader, # 模拟
        test_loader=test_loader
    )
    
    # 4. 加载最佳模型权重
    trainer.model.load_state_dict(torch.load(model_save_path, map_location=trainer.device))
    print(f"成功从 {model_save_path} 加载模型权重。")
    
    # 5. 在测试集上评估
    print("\n--- 在测试集上进行最终评估 ---")
    trainer.model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            text, labels, lengths = [t.to(trainer.device) for t in batch]
            predictions = trainer.model(text, lengths)
            
            preds = predictions.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # 6. 计算详细指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary' # 假设是二分类
    )
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_inference_time_ms = (inference_time / len(test_dataset)) * 1000

    print(f"评估完成。总耗时: {inference_time:.2f} 秒")
    print(f"  - 准确率 (Accuracy): {accuracy*100:.2f}%")
    print(f"  - 精确率 (Precision): {precision:.4f}")
    print(f"  - 召回率 (Recall): {recall:.4f}")
    print(f"  - F1 分数 (F1-Score): {f1:.4f}")
    print(f"  - 平均推理时间: {avg_inference_time_ms:.4f} ms/sample")

    # 7. 保存结果到CSV
    summary_path = "results/summary.csv"
    results_df = pd.DataFrame([{
        'model': model_name,
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'avg_inference_time_ms': round(avg_inference_time_ms, 4)
    }])
    
    if pd.io.common.file_exists(summary_path):
        results_df.to_csv(summary_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(summary_path, mode='w', header=True, index=False)
        
    print(f"\n结果已保存至 {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument('--config', type=str, required=True, help='实验配置文件的路径')
    args = parser.parse_args()
    
    # 示例用法:
    # python src/evaluate.py --config configs/gru.yaml
    evaluate_model(args.config) 