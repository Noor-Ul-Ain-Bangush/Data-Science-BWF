#CHAPTER:03-----BUILT-IN DATA STRUCTURES, FUNCTIONS & FILES
#TUPLE
#exp:01
tup = (4, 5, 6)
print(tup)

#exp:02
tup1 = 4, 5, 6
print(tup1)

#exp:03
tuple([4, 0, 2])
print(tuple)

#exp:04
tup2 = tuple('string')
print(tup2)

#exp:05
print(tup[0])

#exp:06
nested_tup = (4, 5, 6), (7, 8)
print(nested_tup)

#LIST
#exp:01
lst = [1, 2, 3]
print(lst)

#exp:02
lst[1] = 'change'
print(lst[1])

#exp:03
a_list = [2, 3, 7, None]
tup = ("foo", "bar", "baz")
b_list = list(tup)
print(b_list)

#exp:04
b_list[1] = "peekaboo"
print(b_list)

#exp:05
b_list.append("dwarf")
print(b_list)

#exp:06
b_list.insert(1, "red")
print(b_list)

#exp:07
b_list.append("foo")
print(b_list)

#exp:08
b_list.remove("foo")
print(b_list)

#SORTING
#exp:01
a = [7, 2, 5, 1, 3]
a.sort()
print(a)

#SLICING
#exp:01
seq = [7, 2, 3, 7, 5, 6, 0, 1]
seq[1:5]

#Slices can also be assigned with a sequence:
seq[3:5] = [6, 3]
print(seq)

#DICTIONARY
#exp:01
empty_dict = {}
d1 = {"a": "some value", "b": [1, 2, 3, 4]}
print(d1)

#exp:02
d1[7] = "an integer"
print(d1)
print(d1["b"])

#exp:03
tuples = zip(range(5), reversed(range(5)))
print(tuples)

#exp:04
mapping = dict(tuples)
print(mapping)

#exp:05
words = ["apple", "bat", "bar", "atom", "book"]
by_letter = {}
for word in words:
    letter = word[0]
if letter not in by_letter:
    by_letter[letter] = [word]
else:
    by_letter[letter].append(word)

print(by_letter)

#exp:06
hash("string")
hash((1, 2, (2, 3)))
#hash((1, 2, [2, 3])) # fails because lists are mutable

#exp:07
d = {}
d[tuple([1, 2, 3])] = 5
print(d)

#SET
#exp:01
x = set([2, 2, 2, 1, 3, 3])
print(x)

#exp:02
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
x = a.union(b)
print(x)
y = a | b
print(y)

#exp:03
c = a.copy()
c |= b
print(c)

#exp:04
d = a.copy()
d &= b
print(d)

#exp:05
my_data = [1, 2, 3, 4]
my_set = {tuple(my_data)}
print(my_set)

#exp:06
a_set = {1, 2, 3, 4, 5}
z = {1, 2, 3}.issubset(a_set)
print(z)
u = a_set.issuperset({1, 2, 3})
print(u)

#exp:07
s = {1, 2, 3} == {3, 2, 1}
print(s)

#exp:08
v = sorted([7, 1, 2, 6, 0, 3, 2])
print(v)

#exp:09
o = sorted("horse race")
print(o)

#ZIP
#exp:01
seq1 = ["foo", "bar", "baz"]
seq2 = ["one", "two", "three"]
zipped = zip(seq1, seq2)
print(list(zipped))

#exp:02
t = list(reversed(range(10)))
print(t)

#exp:03
strings = ["a", "as", "bat", "car", "dove", "python"]
e = [x.upper() for x in strings if len(x) > 2]
print(e)

#exp:04
loc_mapping = {value: index for index, value in enumerate(strings)}
print(loc_mapping)

#exp:05
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
print(flattened)

#exp:06
def my_function(x, y):
    return x + y
result = my_function(1, 2)
print(result)

#exp:07
gen = (x ** 2 for x in range(100))
print(gen)














