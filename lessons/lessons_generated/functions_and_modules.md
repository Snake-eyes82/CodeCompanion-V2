# Python Functions and Modules: A Beginner's Guide
**Main Concept:** Functions are reusable blocks of code that perform specific tasks. Modules are files containing Python code (functions, classes, variables) that can be imported and used in other programs.
## What is a Function?
A function is a self-contained block of code that performs a specific task.  It helps organize your code, making it more readable, reusable, and easier to debug. Functions take inputs (arguments), process them, and may return an output (result).
## Defining a Function
Functions are defined using the `def` keyword, followed by the function name, parentheses `()`, and a colon `:`.
Arguments are placed within the parentheses.  The code block inside the function is indented.
```python
def greet(name):
    print(f"Hello, {name}!")
greet("Alice")
```
## Return Values
Functions can return values using the `return` statement.  If no `return` statement is present, the function implicitly returns `None`.
```python
def add(x, y):
    return x + y
sum = add(5, 3)
print(sum)
```
## Modules
Modules are files containing Python code. They allow you to organize your code into reusable components and use code written by others. You can import modules using the `import` statement.
## Importing Modules
To use a module, you import it using the `import` statement. For example, to use the `math` module, you would write `import math`. Then, you can access functions and constants within the module using dot notation (e.g., `math.sqrt(2)`).
## Using Modules
Once a module is imported, you can use its functions and variables as if they were part of your current program.
```python
import math
result = math.sqrt(25)
print(result)
```
