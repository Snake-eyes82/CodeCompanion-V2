# File Handling in Python
**Main Concept:** Learn how to interact with files in Python, enabling you to read data from and write data to files.
## Opening and Closing Files
The core of file handling involves opening a file using the `open()` function and closing it afterward using the `close()` method.  The `open()` function takes the file path as the first argument and the mode as the second (e.g., 'r' for reading, 'w' for writing, 'a' for appending). Always remember to close files to prevent data loss or corruption.
## Reading from Files
Once a file is opened in read mode ('r'), you can read its contents using methods like `read()`, `readline()`, and `readlines()`.  `read()` reads the entire file at once, `readline()` reads one line at a time, and `readlines()` reads all lines into a list.
```python
file = open('my_file.txt', 'r')
content = file.read()
print(content)
file.close()
```
## Writing to Files
To write to a file, open it in write mode ('w') or append mode ('a').  'w' overwrites existing content, while 'a' adds new content to the end. Use the `write()` method to write strings to the file.
```python
file = open('my_file.txt', 'w')
file.write('This is some text.')
file.close()
```
## Using with Statement (Context Manager)
The `with` statement simplifies file handling by automatically closing the file even if errors occur. This is the recommended way to work with files.
```python
with open('my_file.txt', 'r') as file:
    content = file.read()
    print(content)
```
## Error Handling
It's crucial to handle potential errors like `FileNotFoundError` when working with files. Use `try...except` blocks to gracefully manage such situations.
```python
try:
    with open('my_file.txt', 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print('File not found.')
```
