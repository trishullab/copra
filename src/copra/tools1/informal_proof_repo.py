#!/usr/bin/env python3

import os
import typing
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class InformalProofRepo:
    repo_map: typing.Dict[str, typing.Tuple[str, str]] = field(default_factory=dict)

    def informal_proof_exists(self, theorem_name: str) -> bool:
        return theorem_name in self.repo_map
    
    def get_informal_thm_proof(self, theorem_name: str) -> typing.Tuple[str, str]:
        return self.repo_map[theorem_name]
    
    def add_informal_thm_proof(self, theorem_name: str, informal_stmt: str, informal_proof: str):
        self.repo_map[theorem_name] = (informal_stmt, informal_proof)
    
    def serialize(self) -> str:
        return self.to_json()
    
    def save(self, file_path: str):
        with open(file_path, "w") as f:
            f.write(self.serialize())
    
    @staticmethod
    def load_from_file(file_path: str) -> "InformalProofRepo":
        assert os.path.exists(file_path), "file_path must be a valid path to a file"
        json_text = None
        with open(file_path, "r") as f:
            json_text = f.read()
        return InformalProofRepo.load_from_string(json_text)

    @staticmethod
    def load_from_string(repo_content: str):
        return InformalProofRepo.schema().loads(repo_content)

if __name__ == "__main__":
    repo = InformalProofRepo()
    repo.add_informal_thm_proof("test", "test", "test")
    print(repo.informal_proof_exists("test"))
    print(repo.get_informal_thm_proof("test"))
    print(repo.informal_proof_exists("test2"))