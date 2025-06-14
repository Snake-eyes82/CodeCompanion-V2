# Python Data Structures: Lists, Tuples, and Dictionaries
**Main Concept:** Understanding and utilizing fundamental Python data structures: lists (mutable, ordered sequences), tuples (immutable, ordered sequences), and dictionaries (key-value pairs).
## Lists
Lists are ordered, mutable (changeable) sequences of items. They can contain items of different data types.  Lists are defined using square brackets `[]`.
```python
my_list = [1, 2, 'apple', 3.14, True]
print(my_list)
my_list.append(5) 
print(my_list)
```
## Tuples
Tuples are ordered, immutable (unchangeable) sequences of items.  They are defined using parentheses `()`. Once created, you cannot modify a tuple.
```python
my_tuple = (1, 2, 'apple', 3.14, True)
print(my_tuple)
# my_tuple.append(5)  # This will cause an error because tuples are immutable
```
## Dictionaries
Dictionaries are collections of key-value pairs. Keys must be immutable (like strings or numbers), and values can be of any data type. Dictionaries are defined using curly braces `{}`.
```python
my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}
print(my_dict)
print(my_dict['name'])
```
## Key Differences
Lists are mutable and use square brackets [], while tuples are immutable and use parentheses (). Dictionaries store key-value pairs and use curly braces {}.  Choose the data structure that best suits your needs based on whether you need mutability and the type of data you are storing.
