# Represents one training sample
from label import Label


class Sample:

    def __init__(self, word, base, category, token, label, specification):
        self.word = word
        self.base = base
        self.category = category
        self.token = token
        self.label = Label(label)
        self.specification = specification

    def __repr__(self):
        return f"{self.word} - {self.token} -> {self.label.text}"
