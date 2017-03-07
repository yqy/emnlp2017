#coding=utf8
import sys

class wordAttrDict():
    def __init__(self,file_path):
        self.word_dict = {}
        f = open(file_path)
        line = f.readline()
        while True:
            line = f.readline()
            if not line:break
            line = line.strip().split("\t")
            word = line[0]
            animate_times = int(line[1])
            inanimate_times = int(line[2])
            male_times = int(line[3])
            female_times = int(line[4])
            neural_times = int(line[5])
            single_times = int(line[6])
            plural_times = int(line[7])

            if (animate_times + inanimate_times + male_times + female_times + neural_times + single_times + plural_times) <= 5:
                continue 
            else:
                animate = 2 # 动物属性 0:动物 1:非动物 2:未知
                sex = 6 #性别属性 3:男性 4:女性 5:中性 6:未知
                number = 9 #数目属性 7:单数 8:复数 9:未知

                if (animate_times + inanimate_times) > 0:
                    if animate_times > inanimate_times:
                        animate = 0
                    else:
                        animate = 1

                if (male_times + female_times + neural_times) > 0:
                    if (male_times > female_times) and (male_times > neural_times):
                        sex = 3
                    elif (female_times > male_times) and (female_times > neural_times):
                        sex = 4
                    else:
                        sex = 5

                if (single_times + plural_times) > 0:
                    if single_times > plural_times:
                        number = 7
                    else:
                        number = 8
                
                self.word_dict[word] = (animate,sex,number)
    def get_att(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return (2,6,9)

def main():
    wa = wordAttrDict("./zh-attributes.data")

    while True:
        line = sys.stdin.readline()
        print wa.get_att(line.strip())                        

if __name__ == "__main__":
    main()
