import numpy
import numpy.linalg as nla

print('ndarray数组与标量/数组的运算')
x = numpy.array([[1, 2, 3], [3, 4, 5]])
print(x * 2)  # [2 4 6]
print(x > 2)  # [False False  True]
y = numpy.array([3, 4, 5])
print(x + y)  # [4 6 8]
print(x > y)  # [False False False]

print('线性代数')

print('矩阵点乘')
x = numpy.array([[1, 2], [3, 4]])
y = numpy.array([[1, 3], [2, 4]])
print(x.dot(y))  # [[ 5 11][11 25]]
print(numpy.dot(x, y))  # [[ 5 11][11 25]]
print('矩阵求逆')
x = numpy.array([[1, 1], [1, 2]])
y = nla.inv(x)  # 矩阵求逆（若矩阵的逆存在）
print(y)
print(x.dot(y))  # 单位矩阵 [[ 1.  0.][ 0.  1.]]
print(nla.det(x))  # 求行列式

print('取对角线上的元素')
x = numpy.array([[1, 0], [0, 1]])
print(numpy.diag(x))  # 取对角线上的元素
x = numpy.array([1, 2])
print(numpy.diag(x))  # 将一维数组转化成矩阵
