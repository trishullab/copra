import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.tools.training_data_format import TrainingDataFormat

class ModelConfig(object):
    def __init__(self, model_name: str, max_length: int, gpu_off: bool, other_params: typing.Dict[str, typing.Any]):
        assert isinstance(model_name, str)
        assert len(model_name) > 0
        assert isinstance(max_length, int)
        assert isinstance(other_params, dict)
        assert isinstance(gpu_off, bool)
        self.model_name = model_name
        self.max_length = max_length
        self.other_params = other_params
        self.gpu_off = gpu_off

class Tactic(object):
    def __init__(self, tatics : typing.List[str]):
        assert isinstance(tatics, list)
        assert all(isinstance(tactic, str) for tactic in tatics)
        assert len(tatics) > 0
        self.tactics = tatics
    
    def __str__(self):
        return "\n".join(self.tactics)

class BaseTacticGenerator(object):
    def __init__(self, model_config: ModelConfig):
        assert isinstance(model_config, ModelConfig)
        self.model_config = model_config

    def generate_tactics(self, data_point: TrainingDataFormat, k: int = 1) -> typing.List[Tactic]:
        """Get the proof tactic(s) for the given data point.

        Args:
            data_point (TrainingDataFormat): The data point for which the tactic is to be generated.

        Returns:
            typing.List[Tactic]: List of tactic(s) for the given data point.
        """
        raise NotImplementedError("This method is not implemented.")