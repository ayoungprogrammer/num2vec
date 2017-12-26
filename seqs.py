import numpy as np


OPS = []

class AST:
    def __init__(self, recursive=False):
        self.recursive = recursive

    def generate_seq(self, seq_len, recursive=False):
        if recursive:
            a = np.random.randint(-10, 10)
            seq = [a]
            for i in range(1, seq_len+1):
                xp = seq[-1]
                scope = {'var': xp}
                seq.append(self.execute(scope))
        else:
            seq = []
            for x in range(1, seq_len+1):
                scope = {'var': x}
                seq.append(self.execute(scope))
        return seq

    @staticmethod
    def generate_ast(max_depth, recursive):
        if max_depth > 0:
            if recursive:
                op = np.random.choice(5, p=[0.05, 0.35, 0.25, 0.25, 0.1])
            else:
                op = np.random.choice(6, p=[0.05, 0.35, 0.15, 0.15, 0.15, 0.15])
            if op == 0:
                return Variable(recursive)
            else:
                return OPS[op](max_depth - 1, recursive)
        else:
            return Variable(recursive)


class Variable(AST):

    def execute(self, scope):
        return scope['var']

    def print(self):
        return 'x'

class Const(AST):
    def __init__(self, min_val=0, max_val=1000):
        self.val = np.random.randint(min_val, max_val+1)

    def execute(self, scope):
        return self.val

    def print(self, recursive=False):
        return self.val

class BinaryOp(AST):
    def __init__(self, max_depth, recursive):
        super(BinaryOp, self).__init__(recursive)
        if np.random.randint(0, 2) == 0:
            self.left = AST.generate_ast(max_depth-1, recursive)
            self.right = self.make_const()
        else:
            self.left = self.make_const()
            self.right = AST.generate_ast(max_depth-1, recursive)

    def make_const(self):
        return Const()

    def op(self, x, y):
        pass

    def execute(self, scope):
        x = self.left.execute(scope)
        y = self.right.execute(scope)
        return self.op(x,y)

class AddOp(BinaryOp):
    def op(self, x, y):
        return x + y

    def print(self):
        return '( %s ) + ( %s )' % (self.left.print(), self.right.print())

class SubtractOp(BinaryOp):
    def __init__(self, max_depth, recursive):
        super(SubtractOp, self).__init__(max_depth, recursive)
        if np.random.randint(0, 2) == 0:
            self.left = AST.generate_ast(max_depth-1, recursive)
            self.right = Const(2, 100)
        else:
            self.left = Const(10, 99999)
            self.right = AST.generate_ast(max_depth-1, recursive)

    def op(self, x, y):
        return x - y

    def print(self):
        return '( %s ) - ( %s )' % (self.left.print(), self.right.print())

class MultiplyOp(BinaryOp):
    def op(self, x, y):
        return x * y

    def make_const(self):
        return Const(-15, 15)

    def print(self):
        return '( %s ) * ( %s )' % (self.left.print(), self.right.print())

class ExpOp(BinaryOp):
    def __init__(self, max_depth, recursive):
        super(ExpOp, self).__init__(max_depth, recursive)
        if np.random.randint(0, 2) == 0:
            self.left = Variable()
            self.right = Const(2, 10)
        else:
            self.left = Const(2, 10)
            self.right = Variable()

    def op(self, x, y):
        return x ** y

    def print(self):
        return '( %s ) ^ ( %s )' % (self.left.print(), self.right.print())

class DivideOp(BinaryOp):
    def __init__(self, max_depth, recursive):
        super(DivideOp, self).__init__(max_depth, recursive)
        self.left = AST.generate_ast(max_depth-1, recursive)
        self.right = Const(2, 10)

    def op(self, x, y):
        return x // y

    def print(self):
        return '( %s ) / ( %s )' % (self.left.print(), self.right.print())


OPS = [Variable, AddOp, SubtractOp, MultiplyOp, DivideOp, ExpOp]


def generate_arithmetic_seq(seq_len):
    root = AST.generate_ast(3, recursive=False)
    ints = root.generate_seq(seq_len)
    return ints, root.print()

def generate_cycle(seq_len):
    cyc_len = np.random.randint(2, seq_len/2+1)
    pat = list(np.random.randint(1, 100, cyc_len))
    ints = sum([pat for x in range(seq_len)], [])
    start = np.random.randint(0, cyc_len)
    return ints[start:start+seq_len], str(pat) + ' repeat'

def generate_recursive(seq_len):
    root = AST.generate_ast(3, recursive=True)
    ints = root.generate_seq(seq_len)
    pat = root.print().replace('x', 'f(x-1)')
    return ints, pat


def generate_seq(seq_len):
    ind = np.random.choice(3, p=[0.1, 0.7, 0.2])
    fn = [generate_cycle, generate_arithmetic_seq, generate_recursive]
    
    while True:
        ints, seq_str = fn[ind](seq_len)
        if max(ints) < 10000000 and min(ints) > 0:
            break
    return fn[ind](seq_len)


def main():
    # root = generate_ast(5)
    # print(root.print())
    # print(root.generate_seq(5))
    with open('test-data.txt', 'w') as f:
        pats = set()
        for i in range(3000):
            while True:
                seq, pat = generate_seq(8)
                if max(seq) < 10000000 and min(seq) > 0 and pat not in pats:
                    break
            pats.add(pat)
            f.write(pat + '\t' + '\t'.join(map(str, seq)) + '\n')


if __name__ == "__main__":
    main()
