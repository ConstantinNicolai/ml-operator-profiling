a = [1,1,1,1,1,1]

b = [0,0,0]


print(a)

print(len(a))



print(b)

print(len((b)))


c = []

c.extend(a)

c.extend(b)
print(c)


print(c[0:len(a)])

print(c[len(a):len(a)+len(b)])