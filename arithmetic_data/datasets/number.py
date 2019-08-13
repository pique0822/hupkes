class Term(object):
    """Represents a term for which we can perform arithmetic operations"""

    def __init__(self):
        super(Term, self).__init__()
        self.numeric_value = None
        self.to_string = ""

    def get_value(self):
        return self.numeric_value

    def __str__(self):
        return self.to_string

class Number(Term):
    """represents a simple number which we can perform arithmetic on"""

    def __init__(self, value):
        super(Number, self).__init__()
        self.numeric_value = value
        if value is not None:
            if value < 0:
                self.sign = '-'
            else:
                self.sign = ''

            number_possibilities = ['zero','one','two','three','four','five','six','seven','eight','nine','ten']

            self.to_string = self.sign+number_possibilities[abs(value)]

class Operation(Term):
    """represents a pair of Terms on which we can perform operations"""

    def __init__(self, left=None, right=None, operation='+'):

        super(Operation, self).__init__()
        self.left = left
        self.right = right
        self.op = operation
        if operation == '+':
            self.text_operation = 'plus'
        elif operation == '-':
            self.text_operation = 'minus'
        elif operation == '*':
            self.text_operation = 'times'
        else:
            self.text_operation = 'dividedBy'

        if (self.right is not None and self.right.get_value() == 'ERROR') or (self.left is not None and self.left.get_value() == 'ERROR'):
            self.numeric_value = 'ERROR'

        if right is not None and left is not None:
            try:
                self.numeric_value = eval(str(self.left.get_value()) + self.op + str(self.right.get_value()))
            except:
                self.numeric_value = 'ERROR'

            self.to_string = '( '+str(self.left) + ' '+ self.text_operation + ' ' + str(self.right)+' )'

            self.recursive_value = '( '+str(self.left.get_value())+operation+str(self.right.get_value())+' )'

        elif left is not None and right is None:
            self.numeric_value = self.left.get_value()
            self.to_string = str(self.left)

            self.recursive_value = '( '+str(self.left.get_value())+' )'
        elif right is not None and left is None:
            self.numeric_value = self.right.get_value()
            self.to_string = str(self.right)
            self.recursive_value = '( '+str(self.right.get_value())+' )'
        else:
            self.numeric_value = 0
            self.to_string = ""
            self.recursive_value = '(0)'


if __name__ == '__main__':
    print('CLASS TESTS')
    num1 = Number(10)
    print('Number 1',num1)
    num2 = Number(-2)
    print('Number 2',num2)
    num3 = Number(0)
    print('Number 3',num3)

    op1 = Operation(num1, num2, '+')
    op2 = Operation(num3, op1, '*')
    op3 = Operation(op2, op2, '-')
    op4 = Operation(op2)

    print(op3)
    print(op4)
