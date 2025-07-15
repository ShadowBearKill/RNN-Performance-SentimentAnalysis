import unittest
import os
import yaml
from src.utils import load_config

class TestUtils(unittest.TestCase):

    def setUp(self):
        """设置测试环境"""
        self.test_dir = 'tests/temp_data'
        os.makedirs(self.test_dir, exist_ok=True)
        self.config_path = os.path.join(self.test_dir, 'config.yaml')
        
        self.config_data = {
            'model': {'name': 'TestModel', 'version': 1},
            'training': {'lr': 0.01}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_data, f)

    def tearDown(self):
        """清理测试环境"""
        os.remove(self.config_path)
        os.rmdir(self.test_dir)

    def test_load_config_success(self):
        """测试成功加载配置文件"""
        config = load_config(self.config_path)
        self.assertEqual(config, self.config_data)

    def test_load_config_file_not_found(self):
        """测试当文件不存在时引发FileNotFoundError"""
        non_existent_path = os.path.join(self.test_dir, 'no_such_file.yaml')
        with self.assertRaises(FileNotFoundError):
            load_config(non_existent_path)

if __name__ == '__main__':
    unittest.main() 