import typing
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class TheoremDetails:
    theorem_name: str
    theorem_namespace: str
    theorem_file_path: str
    theorem_pos: typing.Dict[str, int] = field(default_factory=dict) 

    def __str__(self):
        return f"{self.theorem_namespace}.{self.theorem_name}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, TheoremDetails):
            return False
        return self.theorem_name == other.theorem_name and self.theorem_namespace == other.theorem_namespace and self.theorem_file_path == other.theorem_file_path
    
    def __hash__(self):
        return hash((self.theorem_name, self.theorem_namespace, self.theorem_file_path))