# A set of movie critics and their ratings of a small
# set of movies

from math import sqrt

# Returns a difference based ( Euclidean ) similarity score for person1 and person2

def sim_distance(prefs, person1, person2):
    #Get the list of shared items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
    #If they have no shared interest, return 0
    if len(si) == 0:
        return 0

    #Add up the squares of all the differences
    sum_of_sqares = sum([pow(prefs[person1][item] - prefs[person2][item], 2) for item in prefs[person1] if item in prefs[person2]])

    return 1/(1 + sum_of_sqares)

def sim_pearson(prefs, person1, person2):
    #get the list of the items
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1

    #If they have no shared interest return zero
    n = len(si)
    if n == 0:
        return 0

    # Add up all the preferences
    sum1 = sum([prefs[person1][it] for it in si])
    sum2 = sum([prefs[person2][it] for it in si])

    # Sum of the squares
    sum1Sq = sum([pow(prefs[person1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[person2][it], 2) for it in si])

    # Sum of the products
    pSum = sum([prefs[person1][it] * prefs[person2][it] for it in si])

    # Calculate the Pearson score
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2)/n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num/den
    return r

def topMatches(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, other), other) for other in prefs if other!= person]

    #Sort the scores

    scores.sort()
    scores.reverse()
    return scores[0:n]

def getRecommendations(prefs, person, similarity=sim_pearson):
    # Gets recommendations for a person by using weighted average
    # of every other user's ranking.
    totals = {}
    simSums = {}
    for other in prefs:
        if other == person:
            continue
        sim = similarity(prefs, person, other)

        #ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in prefs[other]:

            #only score movies I have not seen
            if item not in prefs[person] or prefs[person][item] == 0:
                # similarity * score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

    # Create the normalised ranking
    rankings = [(total/simSums[item], item) for item, total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            result[item][person] = prefs[person][item]
    return result


critics = {'Lisa Rose': {'Lady in the water': 2.5, 'Snakes on a plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, me and Dupree': 2.5, 'The Night Listener': 3.0},
    'Gene Seymour': {'Lady in the water': 3.0, 'Snakes on a plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'You, me and Dupree': 3.5, 'The Night Listener': 3.0},
    'Michael Phillips': {'Lady in the water': 2.5, 'Snakes on a plane': 3.0, 'Superman Returns': 3.5, 'The Night Listener': 4.0},
    'Claudia Puig': {'Snakes on a plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0, 'You, me and Dupree': 2.5},
    'Mick LaSalle': {'Lady in the water': 3.0, 'Snakes on a plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, 'You, me and Dupree': 2.0},
    'Jack Matthews': {'Lady in the water': 3.0, 'Snakes on a plane': 4.0, 'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You me and Dupree': 3.5},
    'Toby': {'Snakes on a plane': 4.5, 'You, me and Dupree': 1.0, 'Superman Returns': 4.0}}
