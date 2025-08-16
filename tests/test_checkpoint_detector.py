"""
Checkpoint检测器的单元测试
"""

import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import pytest

from src.checkpoint_detector import CheckpointDetector
from src.export_exceptions import CheckpointValidationError


class TestCheckpointDetector:
    """CheckpointDetector测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.detector = CheckpointDetector()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """测试后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_checkpoint(self, checkpoint_dir: str, name: str = "checkpoint", 
                              include_optional: bool = True) -> str:
        """创建模拟checkpoint目录"""
        checkpoint_path = Path(checkpoint_dir) / name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # 创建必需文件
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none"
        }
        
        with open(checkpoint_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)
        
        # 创建adapter模型文件（足够大的文件用于测试）
        (checkpoint_path / "adapter_model.safetensors").write_bytes(b"mock_model_data" * 100000)  # 约1.3MB
        
        if include_optional:
            # 创建可选文件
            tokenizer_config = {
                "tokenizer_class": "QwenTokenizer",
                "vocab_size": 32000
            }
            
            with open(checkpoint_path / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f)
            
            (checkpoint_path / "tokenizer.json").write_text('{"version": "1.0"}')
            (checkpoint_path / "special_tokens_map.json").write_text('{}')
            (checkpoint_path / "vocab.json").write_text('{}')
            (checkpoint_path / "merges.txt").write_text('')
        
        return str(checkpoint_path)
    
    def test_detect_direct_checkpoint_directory(self):
        """测试检测直接checkpoint目录"""
        checkpoint_path = self.create_mock_checkpoint(self.temp_dir, "direct_checkpoint")
        
        detected_path = self.detector.detect_latest_checkpoint(checkpoint_path)
        assert detected_path == checkpoint_path
    
    def test_detect_latest_checkpoint_in_subdirectories(self):
        """测试在子目录中检测最新checkpoint"""
        # 创建多个checkpoint
        checkpoint1 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-1")
        checkpoint2 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-2")
        checkpoint3 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-3")
        
        # 修改文件时间，使checkpoint-3最新
        import time
        time.sleep(0.1)
        Path(checkpoint3).touch()
        
        detected_path = self.detector.detect_latest_checkpoint(self.temp_dir)
        assert detected_path == checkpoint3
    
    def test_detect_checkpoint_nonexistent_directory(self):
        """测试检测不存在的目录"""
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        
        with pytest.raises(CheckpointValidationError) as exc_info:
            self.detector.detect_latest_checkpoint(nonexistent_dir)
        
        assert "目录不存在" in str(exc_info.value)
    
    def test_detect_checkpoint_no_valid_checkpoints(self):
        """测试没有有效checkpoint的目录"""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(CheckpointValidationError) as exc_info:
            self.detector.detect_latest_checkpoint(str(empty_dir))
        
        assert "未找到有效的checkpoint" in str(exc_info.value)
    
    def test_list_available_checkpoints(self):
        """测试列出可用checkpoint"""
        checkpoint1 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-1")
        checkpoint2 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-2")
        
        checkpoints = self.detector.list_available_checkpoints(self.temp_dir)
        
        assert len(checkpoints) == 2
        assert checkpoint1 in checkpoints
        assert checkpoint2 in checkpoints
    
    def test_validate_checkpoint_integrity_valid(self):
        """测试验证有效checkpoint的完整性"""
        checkpoint_path = self.create_mock_checkpoint(self.temp_dir, "valid_checkpoint")
        
        is_valid = self.detector.validate_checkpoint_integrity(checkpoint_path)
        assert is_valid is True
    
    def test_validate_checkpoint_integrity_missing_files(self):
        """测试验证缺失文件的checkpoint"""
        checkpoint_path = Path(self.temp_dir) / "invalid_checkpoint"
        checkpoint_path.mkdir()
        
        # 只创建adapter_config.json，缺失adapter_model.safetensors
        adapter_config = {"peft_type": "LORA", "task_type": "CAUSAL_LM"}
        with open(checkpoint_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)
        
        is_valid = self.detector.validate_checkpoint_integrity(str(checkpoint_path))
        assert is_valid is False
    
    def test_get_checkpoint_metadata_complete(self):
        """测试获取完整checkpoint的元数据"""
        checkpoint_path = self.create_mock_checkpoint(self.temp_dir, "complete_checkpoint")
        
        metadata = self.detector.get_checkpoint_metadata(checkpoint_path)
        
        assert metadata.path == checkpoint_path
        assert metadata.is_valid is True
        assert metadata.has_adapter_config is True
        assert metadata.has_adapter_model is True
        assert metadata.has_tokenizer is True
        assert metadata.adapter_config is not None
        assert metadata.adapter_config["peft_type"] == "LORA"
        assert len(metadata.validation_errors) == 0
    
    def test_get_checkpoint_metadata_invalid_config(self):
        """测试获取无效配置的checkpoint元数据"""
        checkpoint_path = Path(self.temp_dir) / "invalid_config_checkpoint"
        checkpoint_path.mkdir()
        
        # 创建无效的adapter_config.json
        invalid_config = {
            "peft_type": "INVALID_TYPE",  # 无效类型
            "r": -1,  # 无效值
            "lora_alpha": 0  # 无效值
        }
        
        with open(checkpoint_path / "adapter_config.json", "w") as f:
            json.dump(invalid_config, f)
        
        (checkpoint_path / "adapter_model.safetensors").write_bytes(b"mock_data")
        
        metadata = self.detector.get_checkpoint_metadata(str(checkpoint_path))
        
        assert metadata.is_valid is False
        assert len(metadata.validation_errors) > 0
        assert any("不支持的PEFT类型" in error for error in metadata.validation_errors)
    
    def test_find_best_checkpoint_latest(self):
        """测试查找最新checkpoint"""
        checkpoint1 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-1")
        checkpoint2 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-2")
        
        # 确保checkpoint-2更新
        import time
        time.sleep(0.1)
        Path(checkpoint2).touch()
        
        best_checkpoint = self.detector.find_best_checkpoint(self.temp_dir, "latest")
        assert best_checkpoint == checkpoint2
    
    def test_find_best_checkpoint_largest(self):
        """测试查找最大checkpoint"""
        checkpoint1 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-1")
        checkpoint2 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-2")
        
        # 让checkpoint-1更大
        large_file = Path(checkpoint1) / "large_file.bin"
        large_file.write_bytes(b"x" * 10000)
        
        best_checkpoint = self.detector.find_best_checkpoint(self.temp_dir, "largest")
        assert best_checkpoint == checkpoint1
    
    def test_find_best_checkpoint_smallest(self):
        """测试查找最小checkpoint"""
        checkpoint1 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-1")
        checkpoint2 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-2")
        
        # 让checkpoint-2更大
        large_file = Path(checkpoint2) / "large_file.bin"
        large_file.write_bytes(b"x" * 10000)
        
        best_checkpoint = self.detector.find_best_checkpoint(self.temp_dir, "smallest")
        assert best_checkpoint == checkpoint1
    
    def test_get_checkpoint_summary(self):
        """测试获取checkpoint摘要"""
        checkpoint_path = self.create_mock_checkpoint(self.temp_dir, "summary_checkpoint")
        
        summary = self.detector.get_checkpoint_summary(checkpoint_path)
        
        assert summary["path"] == checkpoint_path
        assert summary["is_valid"] is True
        assert summary["has_adapter_model"] is True
        assert summary["has_adapter_config"] is True
        assert summary["has_tokenizer"] is True
        assert "adapter_config" in summary
        assert summary["adapter_config"]["peft_type"] == "LORA"
        assert summary["adapter_config"]["r"] == 16
    
    def test_compare_checkpoints(self):
        """测试比较多个checkpoint"""
        checkpoint1 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-1")
        checkpoint2 = self.create_mock_checkpoint(self.temp_dir, "checkpoint-2")
        
        # 创建一个无效的checkpoint
        invalid_checkpoint = Path(self.temp_dir) / "invalid"
        invalid_checkpoint.mkdir()
        
        comparison = self.detector.compare_checkpoints([
            checkpoint1, checkpoint2, str(invalid_checkpoint)
        ])
        
        assert comparison["summary"]["total_count"] == 3
        assert comparison["summary"]["valid_count"] == 2
        assert len(comparison["checkpoints"]) == 3
        
        # 检查有效checkpoint
        valid_checkpoints = [cp for cp in comparison["checkpoints"] if cp["is_valid"]]
        assert len(valid_checkpoints) == 2
        
        # 检查无效checkpoint
        invalid_checkpoints = [cp for cp in comparison["checkpoints"] if not cp["is_valid"]]
        assert len(invalid_checkpoints) == 1
    
    def test_checkpoint_with_corrupted_json(self):
        """测试包含损坏JSON文件的checkpoint"""
        checkpoint_path = Path(self.temp_dir) / "corrupted_checkpoint"
        checkpoint_path.mkdir()
        
        # 创建损坏的JSON文件
        with open(checkpoint_path / "adapter_config.json", "w") as f:
            f.write("{ invalid json content")
        
        (checkpoint_path / "adapter_model.safetensors").write_bytes(b"mock_data")
        
        metadata = self.detector.get_checkpoint_metadata(str(checkpoint_path))
        
        assert metadata.is_valid is False
        assert any("格式无效" in error for error in metadata.validation_errors)
    
    def test_checkpoint_file_size_validation(self):
        """测试checkpoint文件大小验证"""
        checkpoint_path = Path(self.temp_dir) / "size_test_checkpoint"
        checkpoint_path.mkdir()
        
        # 创建正常的adapter_config.json
        adapter_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        with open(checkpoint_path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f)
        
        # 创建过小的adapter_model.safetensors文件
        (checkpoint_path / "adapter_model.safetensors").write_bytes(b"tiny")
        
        metadata = self.detector.get_checkpoint_metadata(str(checkpoint_path))
        
        assert metadata.is_valid is False
        assert any("文件过小" in error for error in metadata.validation_errors)


if __name__ == "__main__":
    pytest.main([__file__])