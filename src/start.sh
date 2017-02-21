#python resolution.py -type azp -data ../data/test
#python resolution.py -type azp -data ../data/train

#python resolution.py -type res -data ../data/test > feature.test
#python resolution.py -type res -data ../data/train > feature.train

#python resolution.py -type gold -data ../data/test
#python resolution.py -type gold -data ../data/test > tuning.data
#python resolution.py -type gold -data ../data/test -embedding /Users/yqy/work/data/word2vec/qyyin.cbow > tuning.data.cbow
#python resolution.py -type gold -data ../data/test -embedding /Users/yqy/work/data/word2vec/qyyin.skip
#python resolution.py -type gold -data ../data/test/mz/ -embedding /Users/yqy/work/data/word2vec/qyyin.skip

#python resolution.py -type auto -data ../data/test/mz/ -embedding /Users/yqy/work/data/word2vec/qyyin.skip

#python resolution.py -type template -data ../data/

#python resolution.py -type nn -data ../data/test/ -embedding /Users/yqy/work/data/word2vec/embedding.ontonotes -test_data ../data/test
python resolution.py -type nn_train -data /Users/yqy/Desktop/ACL_2017/data/test -embedding /Users/yqy/work/data/word2vec/embedding.ontonotes -test_data /Users/yqy/Desktop/ACL_2017/data/test > result
