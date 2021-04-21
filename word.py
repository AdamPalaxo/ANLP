# Represents one training sample
from dataclasses import dataclass
from typing import List


@dataclass
class Word:
    """Class represents individual word"""
    text: str
    base: str
    category: str
    token: List[str]
    label: str
    specification: str

    def __repr__(self) -> str:
        """Returns representation of the word"""
        return f"{self.text} - {self.token} -> {self.label}"
