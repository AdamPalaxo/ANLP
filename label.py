# Class label represents label of one sample
class Label:

    def __init__(self, text):
        self.text = text
        self.parts = [x for x in text]

    def __str__(self):
        return f"{self.parts}"

