class Operator(object):
    def __init__(self, opt):
        self.op_type = 'Bool'
        self.operator = opt

    def __str__(self):
        return 'Operator: ' + self.operator


class Comparison(object):
    def __init__(self, opt, left_value, right_value):
        self.op_type = 'Compare'
        self.operator = opt
        self.left_value = left_value
        self.right_value = right_value

    def __str__(self):
        return 'Comparison: ' + self.left_value + ' ' + self.operator + ' ' + self.right_value
