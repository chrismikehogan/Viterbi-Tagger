#!/usr/bin/python

import sys
import math

# icvtag.py
#
# Author: Christopher Hogan
# 12 April 2010

counts_uni = {} # Map of unigram counts
counts_tt = {}  # Map of tt bigram counts
counts_tw = {}  # Map of wt bigram counts
tag_dict = {}   # Map of observed tags for given word
sing_tt = {}    # Map of singletons, sing(.|ti-1)
sing_tw = {}    # Map of singletons, sing(.|ti)

def viterbi(train, test):
    """
    Determines best tag sequence given observations using
    the Viterbi algorithm. Scores sequence against gold.
    """

    count(train)                # Train model parameters
    (obs, gold) = unpack(test)  # Read in test file and tags
    
    neg_infinity = float('-inf')# for logp(0)
    v = {}      # dictionary to store viterbi values
    back = {}   # dictionary to store backpointers
    a = {}      # transition probabilities
    b = {}      # emission probabilities
    progress = 0

    # Initialize for timesteps 0 and 1
    v['0/###']= 1.0
    back['0/###'] = None # This has no effect really
    for tag in tag_dict[obs[1]]:
        v[makekey('1', tag)] = prob('###', tag, 'a') + prob(tag, obs[1], 'b')
        back[makekey('1', tag)] = '###'

    # Recurse
    for j in xrange(2, len(obs)):
        for tj in tag_dict.get(obs[j], tag_dict['OOV']):        # For each possible tag tj for current obs at j

            vj = makekey(str(j), tj)
            for ti in tag_dict.get(obs[j-1], tag_dict['OOV']):  # For each possible previous tag ti leading to current tag tj
                
                vi = makekey(str(j-1), ti)
                tt = makekey(ti, tj)
                tw = makekey(tj, obs[j])

                # If probs are not already known, compute them
                if tt not in a:
                    a[tt] = prob(ti, tj, 'a')
                if tw not in b:
                    b[tw] = prob(tj, obs[j], 'b')

                # then find the viterbi value
                u = v[vi] + a[tt] + b[tw]
    
                if u > v.get(vj, neg_infinity):     # If u is max so far, set it so,
                    v[vj] = u
                    back[makekey(str(j),tj)] = ti   # and store backpointer to ti that gave that u

            # Display progress
            progress += 1
            if progress % 5000 == 1:
                sys.stderr.write('.')

    predict = ['###']
    prev = predict[0]
    known, novel, ktotal, ntotal = 0, 0, 1e-100, 1e-100

    # Follow backpointers to find most likely sequence. As
    # the sequence is built, each tag that is added (except
    # "###") is scored against its annotated tag from gold.
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

    tpct = float(known + novel) / (ktotal + ntotal) * 100
    kpct = float(known) / ktotal * 100
    npct = float(novel) / ntotal * 100
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


    infile.close()
    return words, tags

def count(filename):
    """
count() sets the parameters of the HMM model and
stores them in the maps counts_*, tag_dict, and sing_*
"""

    (words, tags) = unpack(filename)
    progress = 0
    counts_uni['_N_'] = len(tags) - 1 # number of tokens

    # Initialize counts_uni, counts_tw and tag_dict with first timestep and OOV
    tag_dict[words[0]] = [tags[0]]
    tag_dict['OOV'] = []
    
    counts_uni[words[0]] = 1
    counts_uni[tags[0]] = 1
    
    tw = makekey(tags[0], words[0])
    counts_tw[tw] = 1
    sing_tw[tags[0]] = 1

    # Iterate over rest of words/tags
    for i in xrange(1, len(words)):

        tw = makekey(tags[i], words[i])

        # Add all tags except '###' to OOV
        if (tags[i] not in tag_dict['OOV']) and (tags[i] != '###'):
            tag_dict['OOV'].append(tags[i])

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

        # Adjust singleton count
        if (counts_tw[tw] == 1):
            sing_tw[tags[i]] = sing_tw.get(tags[i], 0) + 1
        elif (counts_tw[tw] == 2):
            sing_tw[tags[i]] -= 1

        # Increment unigram counts
        for key in [words[i], tags[i]]:
            counts_uni[key] = counts_uni.get(key, 0) + 1

        # Increment tt count
        tt = makekey(tags[i-1], tags[i])
        counts_tt[tt] = counts_tt.get(tt, 0) + 1

        # Adjust singleton count
        if (counts_tt[tt] == 1):
            sing_tt[tags[i-1]] = sing_tt.get(tags[i-1], 0) + 1
        elif (counts_tt[tt] == 2):
            sing_tt[tags[i-1]] -= 1

        # Display progress
        progress += 1
        if progress % 5000 == 1:
            sys.stderr.write('.')

    counts_uni['_V_'] = len(tag_dict.keys()) # number of types
    
    # Fix unigram counts for "###"
    counts_uni['###'] = counts_uni['###'] / 2
    
    sys.stderr.write("\nFinished training from '%s' on %d tokens\n" % (filename, len(words)))

def prob(i, j, switch):

    # If computing transition probs
    if switch == 'a':
        tt = makekey(i, j)

        backoff = float(counts_uni[j])/counts_uni['_N_']
        lambdap = sing_tt[i] + 1e-100

        return math.log(float(counts_tt.get(tt, 0) + lambdap*backoff)/(counts_uni[i] + lambdap))

    # and if computing emmission
    elif switch == 'b':
        tw = makekey(i, j)

        backoff = float(counts_uni.get(j, 0) + 1)/(counts_uni['_N_']+counts_uni['_V_'])
        lambdap = sing_tw[i] + 1e-100
        return math.log(float(counts_tw.get(tw, 0)+lambdap*backoff)/(counts_uni[i] + lambdap))

    # return prob of 0 if function isnt called properly
    else: return float('-inf')
    
def makekey(*words):
    return '/'.join(words)

def main():
    argv = sys.argv[1:]

    if len(argv) < 2:
        print """
    HMM part-of-speech tagger.
    Determines best (Viterbi) sequence for a given string.

    Usage: %s trainpath testpath
    """ % sys.argv[0]
        sys.exit(1)

    train = argv.pop(0)
    test = argv.pop(0)

    (tpct, kpct, npct, ppw) = viterbi(train, test)
    
    print """
Tagging accuracy: %.4g%% (known: %.4g%% novel: %.4g%%)
Perplexity per tagged test word: %.3f
    """ % (tpct, kpct, npct, ppw)
    
if __name__ == "__main__":
    main()
