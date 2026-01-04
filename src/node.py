# Node
class Node():
    def __init__(self):
        self.value: str | float = None
        self.next: Node = None
        self.childs: list[Node] = None