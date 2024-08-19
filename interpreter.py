# the main is at the very bottom
# at the top of the main you'll find ready test codes


import re
from collections import namedtuple

Token = namedtuple('Token', ['type', 'value'])

# Define the token specification
token_specification = [
    ('FOR', r'\bfor\b'), # "for" keyword
    ('DEF', r'\bdef\b'),    # "def" keyword
    ('RETURN', r'\breturn\b'),  # "return" keyword
    ('PRINT', r'\bprint\b'),    # "print" keyword
    ('LAMBDA', r'\blambda\b'),  # "lambda" keyword
    ('TRUE', r'\bTrue\b'),  # boolean True
    ('FALSE', r'\bFalse\b'),  # boolean False
    ('NUMBER', r'\b\d+\b'),
    ('ID', r'\b[a-z_]\w*\b'),   # identification strings , lowercase only
    ('AND', r'&&'),
    ('OR', r'\|\|'),
    ('NOT', r'!'),
    ('EQ', r'=='),
    ('NEQ', r'!='),
    ('GT', r'>'),
    ('LT', r'<'),
    ('PLUS', r'\+'),
    ('MINUS', r'\-'),
    ('MUL', r'\*'),
    ('DIV', r'\/'),
    ('MODULO', r'%'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('COMMA', r','),
    ('COLON', r':'),
    ('SEMICOLON', r';'),
    ('SKIP', r'[ \t]+'),  # skip spaces and tabs
    ('NEWLINE', r'\n'),  # line endings
    ('MISMATCH', r'.'),  # any other character
]




#################################################
## Position
#################################################
class Position:
    def __init__(self, line_num, char_num):
        self.line_num = line_num
        self.char_num = char_num

    def advance_line(self):
        self.line_num += 1

    def advance_char(self, char_num):
        self.char_num = char_num


#################################################
## Error
#################################################

class Error(Exception):
    def __init__(self, line_num, char):
        self.line_num = line_num
        self.char = char
        super().__init__(f'line: {line_num} char: "{char}"')


class IllegalCharacterError(Error):
    def __init__(self, line_num, char):
        super().__init__(line_num, char)


#################################################
## Lexer
#################################################


class Tokenizer:
    def __init__(self, input_text):
        self.input_text = input_text

    def tokenize(self):
        # Compile the combined regular expression
        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
        get_token = re.compile(tok_regex).match

        # Tokenize the input string
        position = Position(1, 0)

        tokens = []
        mo = get_token(self.input_text)

        while mo is not None:
            typ = mo.lastgroup
            if typ != 'SKIP' and typ != 'MISMATCH':  # if matched sequence inst a whitespace or mismatched character,
                if typ == 'NEWLINE':  # if \n encountered advance to the next line
                    position.advance_line()
                # tokenize it
                val = mo.group(typ)
                if typ == 'NUMBER':
                    val = int(val)
                tokens.append(Token(typ, val))
            else:  # raise illegal character error if unknown character encountered
                if typ == 'MISMATCH':
                    val = mo.group(typ)
                    raise IllegalCharacterError(position.line_num, mo.group(typ))
            position.advance_char(mo.end())  # advance to

            mo = get_token(self.input_text, mo.end())  # advance to next matched sequence or character

        return tokens


#################################################
## AST nodes
#################################################


# general purpose node class for all the keywords
class ASTNode:
    def __init__(self, type_, value=None):
        self.type = type_
        self.value = value
        self.children = []

# each may a set of sub keywords
    def add_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"{self.type}({self.value}, {self.children})"


#################################################
## Parser
#################################################
# parse tokens stream in accordance to the BNF using top-down method
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
        self.current_line = 1  # Track the current line number

    # advance one token ahead, keeping track of newlines '\n'
    def advance(self):
        if self.current_token.type == "NEWLINE":
            self.current_line += 1  # Increment line number on newline
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
            # Skip over any additional NEWLINE tokens
            while self.current_token.type == "NEWLINE":
                self.current_line += 1
                self.pos += 1
                if self.pos < len(self.tokens):
                    self.current_token = self.tokens[self.pos]
                else:
                    break

    # check expected token against the input
    def expect(self, token_type):
        if self.current_token.type == token_type:
            self.advance()
        else:
            raise SyntaxError(
                f' line: {self.current_line}  expected: {token_type}  got: {self.current_token.type}')

    # parse a complex program or a statement
    # form root node of an ast
    def parse_program(self):
        root = ASTNode("Program")
        while self.pos < len(self.tokens):
            if self.current_token.type == "DEF":
                root.add_child(self.parse_function_def())
            elif self.current_token.type == "FOR":
                root.add_child(self.parse_for_loop())
            elif self.current_token.type in {"PRINT", "RETURN", "ID", "LAMBDA"}:
                root.add_child(self.parse_statement())
            else:
                # raise syntax error if unexpected token encountered
                if self.current_token.type != "NEWLINE":  # Ignore NEWLINE tokens
                    raise SyntaxError(
                        f"line: {self.current_line}: Unexpected token: {self.current_token.type}")
                self.advance()
        return root

    # parse a function definition
    def parse_function_def(self):
        node = ASTNode("FunctionDef")
        self.expect("DEF")
        node.add_child(ASTNode("Identifier", self.current_token.value))
        self.expect("ID")
        self.expect("LPAREN")

        if self.current_token.type == "ID":
            node.add_child(self.parse_params())
        self.expect("RPAREN")
        self.expect("LBRACE")
        node.children.extend(self.parse_statements())  # Parse the function body
        self.expect("RBRACE")
        return node

    # parse parameters if there are any
    def parse_params(self):
        node = ASTNode("Params")
        node.add_child(ASTNode("Identifier", self.current_token.value))
        self.expect("ID")
        while self.current_token.type == "COMMA":
            self.expect("COMMA")
            node.add_child(ASTNode("Identifier", self.current_token.value))
            self.expect("ID")
        return node

    # parse key word statements
    def parse_statements(self):
        statements = []
        while self.current_token.type in {"PRINT", "RETURN", "ID", "LAMBDA", "FOR"}:
            if self.current_token.type == "FOR":
                statements.append(self.parse_for_loop())
            else:
                statements.append(self.parse_statement())
            while self.current_token.type == "NEWLINE":  # Skip NEWLINE tokens
                self.advance()
        return statements

    # find appropriate keyword
    def parse_statement(self):
        if self.current_token.type == "PRINT":
            return self.parse_print_statement()
        elif self.current_token.type == "RETURN":
            return self.parse_return_statement()
        elif self.current_token.type == "LAMBDA":
            return self.parse_lambda_function()
        elif self.current_token.type == "ID":
            return self.parse_expression_statement()
        else:
            raise SyntaxError(f"Syntax Error on line {self.current_line}: Unexpected token: {self.current_token.type}")

    def parse_print_statement(self):
        node = ASTNode("PrintStatement")
        self.expect("PRINT")
        self.expect("LPAREN")
        node.add_child(self.parse_expression())
        self.expect("RPAREN")
        self.expect("SEMICOLON")
        return node

    def parse_return_statement(self):
        node = ASTNode("ReturnStatement")
        self.expect("RETURN")
        node.add_child(self.parse_expression())
        self.expect("SEMICOLON")
        return node

    def parse_expression_statement(self):
        node = self.parse_expression()
        self.expect("SEMICOLON")
        return node

    # parse a for loop
    def parse_for_loop(self):
        node = ASTNode("ForLoop")
        self.expect("FOR")
        node.add_child(self.parse_number())  # Parse the loop count
        self.expect("LBRACE")
        node.children.extend(self.parse_statements())  # Parse the loop body
        self.expect("RBRACE")
        return node

    # parse expression which is formed of terms, and they formed of factors
    # expression passes through logical, additive, and multiplication expression from top to down
    # in recursive form to find appropriate binary operator
    def parse_expression(self):
        if self.current_token.type == "LAMBDA":
            return self.parse_lambda_function()
        return self.parse_logical_or_expression()

    def parse_lambda_function(self):
        node = ASTNode("LambdaFunction")
        self.expect("LAMBDA")
        node.add_child(self.parse_params())
        self.expect("COLON")
        node.add_child(self.parse_expression())
        return node

    def parse_logical_or_expression(self):
        node = self.parse_logical_and_expression()
        while self.current_token.type == "OR":
            or_node = ASTNode("LogicalOr")
            self.advance()
            or_node.add_child(node)
            or_node.add_child(self.parse_logical_and_expression())
            node = or_node
        return node

    def parse_logical_and_expression(self):
        node = self.parse_logical_not_expression()
        while self.current_token.type == "AND":
            and_node = ASTNode("LogicalAnd")
            self.advance()
            and_node.add_child(node)
            and_node.add_child(self.parse_logical_not_expression())
            node = and_node
        return node

    def parse_logical_not_expression(self):
        if self.current_token.type == "NOT":
            not_node = ASTNode("LogicalNot")
            self.advance()
            not_node.add_child(self.parse_comparison_expression())
            return not_node
        else:
            return self.parse_comparison_expression()

    def parse_comparison_expression(self):
        node = self.parse_additive_expression()
        while self.current_token.type in {"EQ", "NEQ", "LT", "GT"}:
            comp_node = ASTNode("Comparison", self.current_token.value)
            self.advance()
            comp_node.add_child(node)
            comp_node.add_child(self.parse_additive_expression())
            node = comp_node
        return node

    def parse_additive_expression(self):
        node = self.parse_term()
        while self.current_token.type in {"PLUS", "MINUS"}:
            add_node = ASTNode("Additive", self.current_token.value)
            self.advance()
            add_node.add_child(node)
            add_node.add_child(self.parse_term())
            node = add_node
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.current_token.type in {"MUL", "DIV", "MODULO"}:
            term_node = ASTNode("Term", self.current_token.value)
            self.advance()
            term_node.add_child(node)
            term_node.add_child(self.parse_factor())
            node = term_node
        return node

    def parse_factor(self):
        if self.current_token.type == "NUMBER":
            node = ASTNode("Number", self.current_token.value)
            self.advance()
            return node
        elif self.current_token.type == "ID":
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == "LPAREN":
                return self.parse_function_call()
            else:
                node = ASTNode("Identifier", self.current_token.value)
                self.advance()
                return node
        elif self.current_token.type == "LPAREN":
            self.advance()
            node = self.parse_expression()
            self.expect("RPAREN")
            return node
        elif self.current_token.type in {"TRUE", "FALSE"}:
            node = ASTNode("Boolean", self.current_token.value)
            self.advance()
            return node
        elif self.current_token.type == "NOT":
            node = ASTNode("LogicalNot")
            self.advance()
            node.add_child(self.parse_expression())
            return node
        else:
            raise SyntaxError(f"Syntax Error on line {self.current_line}: Unexpected token: {self.current_token.type}")

    # parse function call
    def parse_function_call(self):
        node = ASTNode("FunctionCall", self.current_token.value)
        self.expect("ID")
        self.expect("LPAREN")
        if self.current_token.type != "RPAREN":
            node.add_child(self.parse_arguments())
        self.expect("RPAREN")
        return node

    # parse arguments
    def parse_arguments(self):
        node = ASTNode("Arguments")
        node.add_child(self.parse_expression())
        while self.current_token.type == "COMMA":
            self.expect("COMMA")
            node.add_child(self.parse_expression())
        return node

    # parse number (for loop iterations)
    def parse_number(self):
        node = ASTNode("Number", self.current_token.value)
        self.expect("NUMBER")
        return node

#################################################
## Interpreter
#################################################
class Interpreter:
    def __init__(self, ast):
        # initialize the interpreter with an ast and an environment
        self.ast = ast
        self.environment = {}

    def interpret(self):
        # start the interpretation process by visiting the root node of the ast
        return self.visit(self.ast)

    def visit(self, node):
        # dynamically call the appropriate method to visit a node based on its type
        method_name = f'visit_{node.type}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        # raise an exception if no visit method for the node type is found
        raise Exception(f'No visit_{node.type} method')

    def visit_Program(self, node):
        # process all the top-level statements in the program
        result = None
        for child in node.children:
            result = self.visit(child)
        return result

    def visit_FunctionDef(self, node):
        # define a function in the environment
        function_name = node.children[0].value
        params = node.children[1]
        body = node.children[2:]
        self.environment[function_name] = (params, body)
        return None

    def visit_LambdaFunction(self, node):
        # create a lambda function with parameters and a body
        params = node.children[0]
        body = node.children[1]
        return (params, body)

    def visit_FunctionCall(self, node):
        # handle function calls, including argument passing and executing the function body
        function_name = node.value
        args = self.visit(node.children[0])  # visit the Arguments node

        if function_name in self.environment:
            params, body = self.environment[function_name]
        else:
            params, body = self.visit(function_name)

        local_env = self.environment.copy()
        for param, arg in zip(params.children, args):
            local_env[param.value] = arg

        previous_env = self.environment
        self.environment = local_env

        result = None
        for statement in body:
            result = self.visit(statement)

        self.environment = previous_env

        return result

    def visit_Arguments(self, node):
        # process the arguments passed to a function
        return [self.visit(arg) for arg in node.children]

    def visit_Params(self, node):
        # retrieve the parameter names for a function
        return [param.value for param in node.children]

    def visit_ReturnStatement(self, node):
        # process a return statement and return its value
        return self.visit(node.children[0])

    def visit_Additive(self, node):
        # handle addition and subtraction operations
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        if node.value == '+':
            return left + right
        elif node.value == '-':
            return left - right

    def visit_Term(self, node):
        # handle multiplication, division, and modulo operations
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        if node.value == '*':
            return left * right
        elif node.value == '/':
            return left / right
        elif node.value == '%':  # evaluate modulo
            return left % right

    def visit_Comparison(self, node):
        # handle comparison operations
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        if node.value == '==':
            return left == right
        elif node.value == '!=':
            return left != right
        elif node.value == '<':
            return left < right
        elif node.value == '>':
            return left > right
        elif node.value == '<=':
            return left <= right
        elif node.value == '>=':
            return left >= right

    def visit_LogicalAnd(self, node):
        # handle logical AND operation
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        return left and right

    def visit_LogicalOr(self, node):
        # handle logical OR operation
        left = self.visit(node.children[0])
        right = self.visit(node.children[1])
        return left or right

    def visit_LogicalNot(self, node):
        # handle logical NOT operation
        operand = self.visit(node.children[0])
        return not operand

    def visit_Identifier(self, node):
        # retrieve the value of a variable or function from the environment
        if node.value in self.environment:
            return self.environment[node.value]
        raise NameError(f"Variable '{node.value}' is not defined.")

    def visit_Number(self, node):
        # process numeric literals
        return int(node.value)

    def visit_Boolean(self, node):
        # process boolean literals (True and False)
        return True if node.value == 'True' else False

    def visit_PrintStatement(self, node):
        # print the result of an expression to the console
        value = self.visit(node.children[0])
        print(value)  # output the result to the console
        return value

    def visit_ForLoop(self, node):
        # process a for loop, executing the loop body the specified number of times
        loop_count = self.visit(node.children[0])  # visit the loop count expression (typically a number)
        loop_body = node.children[1:]  # get the loop body (statements inside the loop)

        for _ in range(loop_count):
            for statement in loop_body:
                self.visit(statement)  # execute each statement in the loop body



#################################################
## Main
#################################################


def main():

# test codes
    code = """
    def myfun(x, y, z) {
        return (x % y) + z;
    }
    print(myfun(10, 3, 2));
    """

    code2 = """
    print(10 > 5);
    print(3 < 2);
    print(10 == 10);
    """

    code3 = """
    def inner(y) {
       return 4*y;
    }
    def outer(x) {
      return x+5;
    }
    print(outer(inner(2)));
    """

    code4 = """
    
    def innerloop (x) {
    
        for 5 {
            print (5 + 1);
        }
    }
    innerloop(2);
    """


    code5 = """
    print((3 > 2 || 5 < 8 ) &&  3 < 9 );
    """



    while True:
        try:
            
            input_text = input(">>> ")
            #input_text = code
            
            
            lexer = Tokenizer(input_text)
            tokens = lexer.tokenize()

            # print all token list
            '''
            for token in tokens:
                print(token)
            print("\n\n\n")
            for token in tokens:
                print(token.value, end='  ') if token.value is not None else print("", end='')
            '''
            parser = Parser(tokens)
            ast = parser.parse_program()
            
            # print AST
            '''
            print(f"\n{ast}")
            '''
            
            
            interpreter = Interpreter(ast)
            interpreter.interpret()
            print("\n\n")
            
            
            
        except IllegalCharacterError as e:
            print(f"Illegal character: {e}")


        except SyntaxError as e:
            print(f"Syntax error encountered: {e}")

        except AttributeError as e:
            print(f"Attribute error encountered")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        #break


if __name__ == "__main__":
    main()
