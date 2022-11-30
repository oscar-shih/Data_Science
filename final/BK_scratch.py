import os
import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import networkx

def readfeaturelist(filename):
    with open(filename) as f:
        out = []        # list of feature names
        for line in f:
            out.append(line.strip())
        return out

def readfeatures(featurefile):
    with open(featurefile) as f:
        out = [] 
        for line in f:
            tokens = line.split()
            profile = {}  # empty profile for the user
            for tok in tokens[1:]:
                feature,val = tok.rsplit(';',1)
                val = int(val)
                if feature not in profile:
                    profile[feature]=[val]
                else:
                    profile[feature].append(val)
            out.append(profile)
        for i in range(len(out)):
            assert out[i]['id'][0] == i  # check that each line was read and placed in the correct place in the list
        return out

def featurematch(profile1, profile2, feature):
    return len(set(profile1[feature]).intersection(set(profile2[feature]))) if feature in profile1 and feature in profile2 else 0

def matchvector(profile1, profile2, featurelist):
    out = []
    for feature in featurelist:
        out.append(featurematch(profile1, profile2, feature))
    return out

def weighteddotproduct(vector1, vector2, weight=None):
    if not weight:
        weight = np.ones(len(vector1))
    return np.inner(vector1, np.multiply(weight, vector2)) / np.mean(weight)

def userfeatures(profile):
    """  Returns a list of the features contained in the user profile """
    return [f for f in profile]

def usermatch(profile1, profile2):
    """ returns the match vector for profile2 using only profile1 features as a reference """
    return matchvector(profile1,profile2,userfeatures(profile1))

def readcircle(userID):
    """
    reads a circle for a given user consisting of circleDD: user1 user2 user3 ...
    and returns a dictionary of the circle['number']=[user1, user2, user3]
    """
    circlefile = './data/Training/'+str(userID)+'.circles'
    with open(circlefile) as f:
        circles = {} 
        for line in f:
            tokens = line.split()
            circleID = int(tokens[0].split('circle')[1].split(':')[0])
            circles[circleID] =[]
            for tok in tokens[1:]:
                circles[circleID].append(int(tok))
        return circles
def charprofile(profilemat):
    out = np.zeros(len(profilemat[0]))
    for row in profilemat:
        for i in range(len(row)):
            out[i] += row[i]
    return out
def display_char_profile(charprofile, featurelist):
    out = []
    for i in range(len(featurelist)):
        if charprofile[i] != 0: 
            out.append(str(featurelist[i]) + ": " + str(charprofile[i]))
    return out

def BK_scratch_outputfile(userid, ref_profile, circles, profile, features, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        print('User: ', userid, "\n", file=f)
        char_profile = {}
        for circle in circles:
            print('Circle:', circle, file=f)
            matchmatrix = [
                matchvector(ref_profile, profile[user], features) for user in circles[circle] 
            ]
            char_profile[circle] = charprofile(matchmatrix)
            out = display_char_profile(char_profile[circle], features)
            for char in out:
                print(char, file=f)
            print("\n", file=f)


if __name__ == '__main__':
    data_dir = "./data"
    features = readfeaturelist(os.path.join(data_dir, 'featureList.txt'))
    profile = readfeatures(os.path.join(data_dir, 'features.txt'))
    userid = 345
    out_path = "./BK_scratch.txt"
    ref_profile = profile[userid]
    circles = readcircle(userid)
    BK_scratch_outputfile(
        userid=userid,
        ref_profile=ref_profile,
        circles=circles,
        profile=profile,
        features=features,
        out_path=out_path
    ) # Generate file called "BK_scratch.txt"
    
