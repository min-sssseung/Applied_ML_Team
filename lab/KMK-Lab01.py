### (C) Python Basics ###

# (a) Variables and assignment
a = 1.0 # double
b = 2 # integer
c = True
s = 'text'

# check the data type
print(a, type(a))
print(b, type(b))
print(c, type(c))
print(s, type(s))

# 1.0 <class 'float'>
# 2 <class 'int'>
# True <class 'bool'>
# text <class 'str'>

# (b) Code block (indentation)
for i in range(4):
    print(i)
    if i > 2:
        print("too much")
print("done")

# 수정 이전 코드는 "done"이 4번 반복 생성함 
# 0
# 1
# 2
# 3
# too much
# done

# (c) Operators
x = 7//2
print(x, type(x))
x = 7/2
print(x, type(x))
x = 7%2
print(x, type(x))
i = 1
i *= 3
print(i)
print(i > 1)
print(i == 1)
print(0 < i < 2)

# 3 <class 'int'>
# 3.5 <class 'float'>
# 1 <class 'int'>
# 3
# True
# False
# False

### (D) Functions ###
# (a) two parameters and one return value
def add(x, y):
    z = x + y
    return z
recv = add(4,5)
print(recv)
# 9

# (b) default value(s)
def multiply(x, y=2):
    return x*y
r1 = multiply(4)
print(r1)
r2 = multiply(4,3)
print(r2)
# 8
# 12
# r1은 사용자가 선언하지 않아 기본값인 y=2로 계산됐고, r2는 사용자가 지정한 y=3으로 계산됐다 

### (E) Lists ###
# (a) list creation
e = [] # empty list
l = [1.0, 2, "a"] # contains 3 elements of different type
print(l)
m = [1, 2, 3, 4, 5] # contains 5 elements of the same type
print(m)
n = list(range(5))
print(n)
# [1.0, 2, 'a']
# [1, 2, 3, 4, 5]
# [0, 1, 2, 3, 4]

# (b) indexing and slicing
print(l[0], m[1], m[-1])
m[2] = -10
print(m)
del m[2]
print(m)
print(m[1:3], m[:], m[:2])
# 1.0 2 5
# [1, 2, -10, 4, 5]
# [1, 2, 4, 5]
# [2, 4] [1, 2, 4, 5] [1, 2]

# (c) combining lists
a = [1, 2, 3]
a.append(4)
print(a)
b = a + [5, 6]
print(b)
a.append([5, 6])
print(a)
# [1, 2, 3, 4]
# [1, 2, 3, 4, 5, 6]
# [1, 2, 3, 4, [5, 6]]

# (d) creating lists in a loop
squares = []
for i in range(5):
    squares.append(i**2)
print(squares)
# [0, 1, 4, 9, 16]

# (e) list methods
l = [1, 9, 7, 3]
l.sort()
print(l)
l = [5, 7, 2, 4]
l1 = l.sort()
print(l1)
print(sorted(l))
print(l)
# [1, 3, 7, 9]
# None
# [2, 4, 5, 7]
# [2, 4, 5, 7]

""" II. Control Flows """
### (A) if-elif-else ###
x = eval(input("Enter a number: "))
if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")
# Enter a number: 3
# positive-1
# Enter a number: -1
# negative
# Enter a number: 0
# zero

### B) if-else uses pass keyword ###
x = eval(input("Enter a number: ")) 
if x < 0:
    pass
else:
    print("non-negative")   
# Enter a number: -1
# Enter a number: 10
# non-negative

### (C) one-line if-else ###
x = eval(input("Enter a number: "))
a = 1 if x > 10 else 0
print ("The value of a is: " + str(a))
# Enter a number: 10
# The value of a is: 0

### (D) for-loops ###
# (a) 
for i in range(3):
    print(i)
# 0
# 1
# 2
# (b) 
for i in [1, 'a', True, 2.3]:
    print(i)
# 1
# a
# True
# 2.3

### (E) while-loops ####
# (a) 
i = 0
while i < 4:
    i += 1
    print(i)
else:
    print("No Break\n")
# 1
# 2
# 3
# 4
# No Break

# # (b) 
i = 0
while i < 4:
    i += 1
    print(i)
    break
else:
    print("Never Executed")
# 1

# (c) 
count = 0
while True: # Infinite loop
    count += 1 # Increment the counter
    print(f"Count is {count}")
    if count == 10:
        break # If count becomes 10, exit the loop
print("The loop has ended.")
# Count is 1
# Count is 2
# Count is 3
# Count is 4
# Count is 5
# Count is 6
# Count is 7
# Count is 8
# Count is 9
# Count is 10
# The loop has ended.

""" III. Libraries (Modules) """
### (A) import a module and use functions of the module ### 
import math
print(math.sqrt(2)) # module_name.function
# 1.4142135623730951

### (B) import the necessary functions, and use them without the prefix ###
from math import sqrt, sin, pi
print(sqrt(2))
print(sin(pi/2))
# 1.4142135623730951
# 1.0

### (C) use alias ###
import math as m
print(m.sqrt(2))
print(m.sin(m.pi/2))
# 1.4142135623730951
# 1.0

""" IV. File Handling """
### (A) ###
file = open('infile.txt', 'r')
for each in file:
    print (each)
file.close()
# Hello
# 
# My name is Minkyu
# 
# studying psychology and data-science

### (B) ###
file = open('infile.txt', 'r')
print (file.read())
# Hello
# My name is Minkyu
# studying psychology and data-science

### (C) ###
with open('infile.txt', 'r') as file:
    data = file.readlines()
for line in data:
    word = line.split()
    print (word)
# ['Hello']
# ['My', 'name', 'is', 'Minkyu']
# ['studying', 'psychology', 'and', 'data-science']

### (D) ###
file = open('./outfile.txt','w')
file.write("This is the write command")
file.write("It allows us to write in a particular file")
file.close()

### (E) ###
with open('./outfile.txt', 'w') as f:
    f.write("Autumn is Coming")

### (F) ###
file = open('./outfile.txt','a')
file.write("This will add new line")
file.close()

""" V. Exception Handling """
### (A) Test some error cases ###
value = 10000

a = value/0
print(a)
# Exception has occurred: ZeroDivisionError
# division by zero
#   File "C:\Users\kmk45\OneDrive\바탕 화면\Applied ML\Lab01.py", line 294, in <module>
#     a = value/0
# ZeroDivisionError: division by zero

x = 10
y = 'Text'
z = x+y
# Exception has occurred: TypeError
# unsupported operand type(s) for +: 'int' and 'str'
#   File "C:\Users\kmk45\OneDrive\바탕 화면\Applied ML\Lab01.py", line 304, in <module>
#     z = x+y
# TypeError: unsupported operand type(s) for +: 'int' and 'str'

### (B) ###
value = 10000
try:
    a = value / 0
    print(a)
except:
    print("An error occurred!")
# An error occurred!

x = 10
y = 'Text'
try:
    z = x + y
except TypeError:
    print("Error: cannot add an int and a str")
# Error: cannot add an int and a str