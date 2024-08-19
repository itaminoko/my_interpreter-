# my_interpreter-

Specifications:


1)	All text inputted must be lowercase or the required symbols 
2)	Each statement must end in semicolon
3)	For something to be seen “print” statement must be used 
4)	No strings
5)	No inner function definitions 
6)	For loop works  only with number parameters
7)	For more specifications see BNF down below





Keywords:
1)	<def> – define a function 
2)	<print> - output to the console
3)	<for> - for loop 
4)	<return> - return value from a function 
5)	<id> - function call







BNF:


<program> ::= <function_def>* <statement>* 

<function_def> ::= "def" <identifier> "(" <params>? " )" "{" <statement>* "}"
<lambda_fun> ::= "lambda" <params>? ":" <statement>
<params> ::= <identifier> | <identifier> "," <params>

<statement> ::=  <for_loop> | <print_statement>  | <return_statement>   | <expression_statement>

<for_loop> ::= "for" <identifier> "in" <expression> "{" <statement>* "}"
<print_statement> ::= "print" "( " <expression> ")" ";"
<return_statement> ::= "return" <expression>? ";"
<expression_statement> ::= <expression> ";"

<expression> ::= <logical_or_expression>

<logical_or_expression> ::= <logical_and_expression> 
                          | < logical_ or_expression> "||" <logical_and_expression>
<logical_and_expression> ::= <logical_not_expression> 
                           | <logical_and_expression> "&&" <logical_not_expression>
<logical_not_expression> ::= <comparison_expression> 
                           | "!" <logical_not_expression >

<comparison_expression> ::= <additive_expression> 
                          | <comparison_expression> <comp_op> <additive_expression>

<comp_op> ::= "==" | "!=" | "<" | "<=" | ">" | ">="

<additive_expression> ::= <term> 
                        | <additive_expression> <add_op> <term>
<term> ::= <factor> 
         | <term> <mul_op> <factor>
<factor>  ::= <number> |  <identifier> | <function_call>  | "(" <expression> “)”  | <boolean>

<function_call> ::=  <identifier> "(" <arguments>? ")"
<arguments> ::= <expression> | <expression> "," <arguments>

<add_op> ::= "+" | "-"
<mul_op> ::= "*" | "/"

<identifier> ::= <letter> <identifier_tail>?
<identifier_tail> ::= <letter_or_digit> |  <letter_or_digit> <identifier_tail>?
<letter> ::= "a" | "b" | ... | "z" 
<letter_or_digit> ::= <letter>  | <digit>
<digit> ::= "0" | "1" | ... | "9"
<number> ::=  <digit> | <digit> <number_tail>
<number_tail> ::= <digit>  | <digit> <number_tail>?

<boolean> ::= "True" | "False"
