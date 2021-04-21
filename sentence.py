from dataclasses import dataclass
from typing import List


@dataclass
class Sentence:
    """Class representing one processed sentence."""
    text: str
    tokens: List[str]
    labels: List[str]

    def get_text(self) -> str:
        """Returns full text of the sentence"""
        return self.text

    def get_tokens(self) -> List[str]:
        """Returns individual tokens of the sence"""
        return self.tokens

    def get_labels(self) -> List[str]:
        """Returns labels of the sentence"""
        return self.labels

    def get_labels_decomposed(self) -> List[List[str]]:
        """Returns list od decomposed sentence labels"""
        return [list(label) for label in self.labels]

    def __repr__(self) -> str:
        """Returns representation of the sentence"""
        return f"{self.text}"
