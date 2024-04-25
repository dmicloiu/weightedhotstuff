import random

def generateRandomMatrix(n, low, high):
    M = []
    for i in range(n):
        l = []
        for j in range(n):
##            if j == 0:
##                l.append(0)
##                continue
            if i != j: 
                l.append(random.randint(low,high))
            else:
                l.append(0)
        M.append(l)
    return M

def printMatrix(matrix):
    columns = len(matrix)
    lines = len(matrix[0])
    result = "[\n"
    for column in range(columns):
        result = result + "["
        for line in range(lines):
            result = result+str(matrix[column][line])
            if line != lines-1:
                result = result + ", "
        if column != columns-1:
            result = result+"],\n"
        else:
            result = result+"]"
        
    return result+"\n]\n"
