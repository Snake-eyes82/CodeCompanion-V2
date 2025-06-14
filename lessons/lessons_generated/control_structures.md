# Control Structures: If Statements and Loops
**Main Concept:** Control structures allow you to control the flow of execution in your Python programs.  This lesson covers `if` statements for conditional execution and loops (`for` and `while`) for repetitive tasks.
## If Statements
If statements execute a block of code only if a certain condition is true.  Python uses indentation to define code blocks.  You can also use `elif` (else if) and `else` to handle multiple conditions.
## Example: If Statement
```python
age = 20
if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")
```
## For Loops
For loops iterate over a sequence (like a list, tuple, or string) or other iterable object.  They are useful when you know the number of iterations in advance.
## Example: For Loop
```python
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```
## While Loops
While loops repeat a block of code as long as a condition is true.  Be careful to avoid infinite loops by ensuring the condition eventually becomes false.
## Example: While Loop
```python
count = 0
while count < 5:
    print(count)
    count += 1
```
## Nested Loops
You can nest loops inside each other to perform more complex iterations.  For example, you could use nested loops to iterate over rows and columns of a matrix.
## Example: Nested Loops
```python
for i in range(3):
    for j in range(2):
        print(f"({i}, {j})")
```
## Break and Continue Statements
`break` exits the loop prematurely, while `continue` skips the current iteration and proceeds to the next.
## Example: Break and Continue
```python
for i in range(10):
    if i == 5:
        break  # Exits the loop when i is 5
    if i % 2 == 0:
        continue # Skips even numbers
    print(i)
```
