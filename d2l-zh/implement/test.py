class MyClass:
    def __init__(self, tokens):
        self.tokens = tokens
    def __getitem__(self, token):
        return self.tokens.get(token)
tokens = {'the':2261, 'time':200, 'machine':85, 'by':103}
obj = MyClass(tokens)
print(obj['the'])