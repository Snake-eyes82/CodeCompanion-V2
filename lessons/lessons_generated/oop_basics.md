# Object-Oriented Programming Basics in Python
**Main Concept:** Object-oriented programming (OOP) is a programming paradigm that organizes code around objects, which contain data (attributes) and methods (functions) that operate on that data.  This approach promotes code reusability, modularity, and maintainability.
## Classes and Objects
A class is a blueprint for creating objects.  It defines the attributes and methods that objects of that class will have. An object is an instance of a class. Think of a class as a cookie cutter and objects as the cookies it creates.
## Attributes
Attributes are variables that hold data within an object.  They represent the characteristics of the object.  For example, a 'Dog' object might have attributes like 'name', 'breed', and 'age'.
## Methods
Methods are functions that are defined within a class. They operate on the object's data (attributes). For example, a 'Dog' object might have a method called 'bark()' that prints "Woof!"
## Example: A Simple Dog Class
```python
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def bark(self):
        print("Woof!")

my_dog = Dog("Buddy", "Golden Retriever")
print(my_dog.name)  # Output: Buddy
my_dog.bark()      # Output: Woof!
```
## Constructors (__init__)
The `__init__` method is a special method called a constructor. It's automatically called when you create a new object.  It's used to initialize the object's attributes.
## Self
The `self` parameter in methods refers to the instance of the class (the object itself).  It's how methods access and modify the object's attributes.
