#!/usr/bin/python

import sys
import math

# icvtag.py
#
# Author: Christopher Hogan
# 12 April 2010

# Define global variables
counts_uni = {} # Map of unigram counts
counts_tt = {}  # Map of tt bigram counts
counts_tw = {}  # Map of wt bigram counts
tag_dict = {}   # Map of observed tags for given word

def viterbi(train, test):
    """
    Determines best tag sequence given observations using
    the Viterbi algorithm. Scores sequence against gold.
    """

    count(train)
    (obs, gold) = unpack(test)

    neg_infinity = float('-inf')
    v = {}
    back = {}
    a = {}
    b = {}

    # Initialize for timesteps 0 and 1
    v['0/###']= 1.0
    back['0/###'] = None
    for tag in tag_dict[obs[1]]:
        v[makekey('1', tag)] = prob('###', tag, 'a') + prob(tag, obs[1], 'b')
        back[makekey('1', tag)] = '###'

    # Recurse
    for j in xrange(2, len(obs)):               # For each timestep after 1
        for tj in tag_dict[obs[j]]:             # For each possible tag tj for current obs at j
            
            vj = makekey(str(j), tj)            # Viterbi key for this timestep and tag
            for ti in tag_dict[obs[j-1]]:       # For each possible previous tag ti leading to current tag tj
                
                vi = makekey(str(j-1), ti)
                tt = makekey(ti, tj)
                tw = makekey(tj, obs[j])

                # If probs arent already known, compute them
                if tt not in a:
                    a[tt] = prob(ti, tj, 'a')
                if tw not in b:
                    b[tw] = prob(tj, obs[j], 'b')
                
                u = v[vi] + a[tt] + b[tw]
                
                if u > v.get(vj, neg_infinity):     # If mu is max so far, set it so,
                    v[vj] = u
                    back[makekey(str(j),tj)] = ti   # and store backpointer for ti that gave that value

    predict = ['###']
    prev = predict[0]
    known, novel, ktotal, ntotal = 0, 0, 1e-100, 1e-100

    # Follow backpointers to find most likely sequence. As
    # the sequence is built, each tag taht is added (except
    # '###') is scored against its annotated tag from gold.
    for i in xrange(len(obs)-1, 0, -1):
        if obs[i] != '###':
            if obs[i] in counts_uni:
                ktotal += 1
                if predict[0] == gold[i]:
                    known += 1
            else:
                ntotal += 1
                if predict[0] == gold[i]:
                    novel += 1
        tag = back[makekey(str(i), prev)]
        predict.insert(0, tag)
        prev = tag
    
    tpct = float(known+novel)/(ktotal+ntotal) * 100
    kpct = float(known)/ktotal * 100
    npct = float(novel)/ntotal * 100
    path_prob = v[makekey(str(len(obs)-1), predict[-1])]
    ppw = math.exp(float(-1*path_prob)/(len(obs)-1))

    return tpct, kpct, npct, ppw

def unpack(filename): # Returns a list of words and parallel list of tags

    try:
        infile = open(filename, 'r')

        tags = []
        words = []

        for line in infile:
            (word, tag) = line.strip().split('/')
            tags.append(tag)
            words.append(word)
    
    except IOError, err:
        sys.exit("Couldn't open file at %s" % (filename))

    finally:
        infile.close()
        return words, tags

def count(filename): # Counts frequencies

    (words, tags) = unpack(filename)

    if len(words) != len(tags):
        sys.exit("Error: word and tag lists of different size")


    # Initialize counts_uni and tag_dict with first word/tag
    tag_dict[words[0]] = [tags[0]]
    counts_uni[words[0]] = 1
    counts_uni[tags[0]] = 1
    tw = makekey(tags[0], words[0])
    counts_tw[tw] = 1

    # Iterate over rest of words/tags
    for i in xrange(1, len(words)):

        tw = makekey(tags[i], words[i])

        # If word/tag bigram has never been observed and
        # is not in tag_dict, add it. Otherwise, append
        # tag to list of possible tags for the word
        if counts_tw.get(tw, 0) == 0:

            if words[i] not in tag_dict:
                tag_dict[words[i]]= [tags[i]]
            else:
                tag_dict[words[i]].append(tags[i])

        # Increment tw count
        counts_tw[tw] = counts_tw.get(tw, 0) + 1

        # Increment unigram counts
        for key in [words[i], tags[i]]:
            counts_uni[key] = counts_uni.get(key, 0) + 1

        # Increment tt count
        
        tt = makekey(tags[i-1], tags[i])
        counts_tt[tt] = counts_tt.get(tt, 0) + 1

    # Fix unigram counts for "###"
    counts_uni['###'] = counts_uni['###'] / 2

def prob(i, j, switch):

    # If computing transition probs
    if switch == 'a':
        tt = makekey(i, j)
        return math.log(float(counts_tt[tt])/counts_uni[i])
    
    # and if computing emmission
    elif switch == 'b':
        tw = makekey(i, j)
        return math.log(float(counts_tw[tw])/counts_uni[i])

    # return prob of 0 if function isnt called properly
    else: return float('-inf')    

def makekey(*words):
    return '/'.join(words)    
    
def main():
    argv = sys.argv[1:]

    if len(argv) < 2:
        print """
Unsmoothed HMM tagger.
Determines best (Viterbi) sequence for a given string.

Usage: %s trainpath testpath
""" % sys.argv[0]
        sys.exit(1)

    train = argv.pop(0)
    test = argv.pop(0)

    (tpct, kpct, npct, ppw) = viterbi(train, test)

    print """
Tagging accuracy: %.4g%% (known: %.4g%% novel: %.4g%%)
Perplexity per tagged test word: %.2f
""" % (tpct, kpct, npct, ppw)
    
if __name__ == "__main__":
    main()
