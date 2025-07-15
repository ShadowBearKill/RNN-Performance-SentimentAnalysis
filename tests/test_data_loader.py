import unittest
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from src.data_loader import (
    load_and_preprocess_data, 
    Vocabulary, 
    vectorize_data,
    SentimentDataset,
    collate_fn
)

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """在每个测试方法运行前设置测试环境"""
        self.test_dir = 'tests/temp_data'
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file_path = os.path.join(self.test_dir, 'test_data.csv')
        
        # 创建一个包含各种情况的虚拟数据集
        dummy_data = {
            'label': [1, 0, 1],
            'review': [
                '环境很好，服务周到！',  # 正常情况
                '房间有...异味，设施##陈旧。', # 包含特殊字符
                '  '  # 空白评论
            ]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(self.test_file_path, index=False)

    def tearDown(self):
        """在每个测试方法运行后清理测试环境"""
        os.remove(self.test_file_path)
        os.rmdir(self.test_dir)

    def test_load_and_preprocess_data(self):
        """测试数据加载和预处理函数"""
        processed_data = load_and_preprocess_data(self.test_file_path)

        # 预期结果
        expected_tokens_1 = ['环境', '很', '好', '服务', '周到']
        expected_tokens_2 = ['房间', '有', '异味', '设施', '陈旧']
        
        # 断言处理后的数据条数（应忽略空评论）
        self.assertEqual(len(processed_data), 2)

        # 断言第一条数据的分词结果和标签
        self.assertEqual(processed_data[0][0], expected_tokens_1)
        self.assertEqual(processed_data[0][1], 1)

        # 断言第二条数据的分词结果和标签
        self.assertEqual(processed_data[1][0], expected_tokens_2)
        self.assertEqual(processed_data[1][1], 0)

    def test_vocabulary_and_vectorization(self):
        """测试词典构建和数据向量化功能"""
        # 准备数据
        processed_data = [
            (['我', '爱', '北京'], 1),
            (['北京', '是', '首都'], 1),
            (['一个', '未知词'], 0)
        ]
        all_tokens = [item[0] for item in processed_data]

        # 1. 测试词典构建
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(all_tokens)

        # 断言特殊token和词典大小
        self.assertIn('<PAD>', vocab.word2idx)
        self.assertIn('<UNK>', vocab.word2idx)
        # 2个特殊token + 6个唯一词
        self.assertEqual(len(vocab), 8)
        self.assertIn('北京', vocab.word2idx)

        # 2. 测试向量化
        vectorized_data = vectorize_data(processed_data, vocab)
        
        # 预期向量化结果
        # '我' -> 2, '爱' -> 3, '北京' -> 4
        expected_vector_1 = [vocab.word2idx['我'], vocab.word2idx['爱'], vocab.word2idx['北京']]
        # '北京' -> 4, '是' -> 5, '首都' -> 6
        expected_vector_2 = [vocab.word2idx['北京'], vocab.word2idx['是'], vocab.word2idx['首都']]
        # '一个' -> 7, '未知词' -> '<UNK>' -> 1
        expected_vector_3 = [vocab.word2idx['一个'], vocab.word2idx['<UNK>']]
        
        self.assertEqual(vectorized_data[0][0], expected_vector_1)
        self.assertEqual(vectorized_data[1][0], expected_vector_2)
        self.assertEqual(vectorized_data[2][0], expected_vector_3)

    def test_dataset_and_dataloader(self):
        """测试PyTorch Dataset和DataLoader的功能"""
        # 准备数据
        vectorized_data = [
            ([2, 3, 4], 1),         # len 3
            ([4, 5, 6, 7], 1),      # len 4
            ([8, 9], 0),            # len 2
            ([10, 11, 12, 13, 14], 0) # len 5
        ]
        
        # 1. 测试Dataset
        dataset = SentimentDataset(vectorized_data)
        self.assertEqual(len(dataset), 4)
        seq, label = dataset[0]
        self.assertTrue(torch.is_tensor(seq))
        self.assertTrue(torch.is_tensor(label))

        # 2. 测试DataLoader
        data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        
        # 获取第一个批次
        batch_seqs, batch_labels, batch_lengths = next(iter(data_loader))
        
        # collate_fn会按长度降序排序
        # 第一个批次的原始数据应该是 ([4, 5, 6, 7], 1) 和 ([2, 3, 4], 1)
        
        # 断言类型
        self.assertTrue(torch.is_tensor(batch_seqs))
        self.assertTrue(torch.is_tensor(batch_labels))
        self.assertTrue(torch.is_tensor(batch_lengths))

        # 断言形状
        self.assertEqual(batch_seqs.shape, torch.Size([2, 4])) # (batch_size, max_len)
        self.assertEqual(batch_labels.shape, torch.Size([2]))
        self.assertEqual(batch_lengths.shape, torch.Size([2]))
        
        # 断言内容
        expected_seqs = torch.tensor([[4, 5, 6, 7], [2, 3, 4, 0]], dtype=torch.long)
        expected_labels = torch.tensor([1, 1], dtype=torch.long)
        expected_lengths = torch.tensor([4, 3], dtype=torch.long)

        self.assertTrue(torch.equal(batch_seqs, expected_seqs))
        self.assertTrue(torch.equal(batch_labels, expected_labels))
        self.assertTrue(torch.equal(batch_lengths, expected_lengths))


if __name__ == '__main__':
    # 这使得测试可以直接通过 `python tests/test_data_loader.py` 运行
    unittest.main() 