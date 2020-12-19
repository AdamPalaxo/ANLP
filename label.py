# Class label represents label of one sample
class Label:

    def __init__(self, text):
        self.text = text
        self.parts = [x for x in text]

    def __repr__(self):
        return f"{self.parts}"

    def get_parts(self):
        return {k: v for (k, v) in enumerate(self.parts)}