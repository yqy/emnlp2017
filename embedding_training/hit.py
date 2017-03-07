import random
import sys

line = sys.stdin.readline()

all_num = 0
words = []
while True:
    line = sys.stdin.readline()
    if not line:break
    line = line.strip().split(" ")
    all_num += int(line[1])
    words += ([line[0]]*int(line[1]))
print all_num

print words[random.randint(0,all_num-1)]
print words[random.randint(0,all_num-1)]
print words[random.randint(0,all_num-1)]
print words[random.randint(0,all_num-1)]
