from scipy.sparse import data
import spacy
import heapq
import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

dat = []


def backend(query):
    nlp = spacy.load('en_core_web_lg')

    liststr = []
    songnamelist = []

    with open('14.csv') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            text = "".join(row[3]).split()
            text = [word.lower() for word in text]
            inputstr = ' '.join(text)
            liststr.append(inputstr)

            songname = ''.join(row[1]).split()
            songnamel = ' '.join(songname)
            songnamelist.append(songnamel)

    lyricstr = []

    customize_stop_words = ['[Verse]', '[Chorus]', '[Pre-Chorus]',
                            ',', '"', '[', ']', 'chorus', '-', '?', '(', ')', ':', '.', '!']

    for w in customize_stop_words:
        nlp.vocab[w].is_stop = True

    for i in liststr:
        doc = nlp(i)
        tokens = [token.lemma_ for token in doc if not token.is_stop]
        lyricstr.append(tokens)

    vectors = np.array([nlp(" ".join(lyric)).vector for lyric in lyricstr])

    templove = ['heart', 'lovely', 'family', 'caring', 'forever', 'trust', 'passion', 'romance', 'sweet', 'kiss',
                'love', 'hugs', 'warm', 'fun', 'kisses', 'joy', 'friendship', 'marriage', 'husband', 'wife', 'forever']
    templove = ' '.join(templove)
    temphappy = ["cheerful", 'happy', "content", "delighted", "ecstatic", "elated", "glad", "joyful", "joyous", "joy", "jubilant", "lively", "merry", "overjoyed", "peaceful", "pleasant", "pleased", "thrilled", "upbeat", "blessed", "blest", "blissful",
                 "blithe", "captivated", "chipper", "chirpy", "convivial", "exultant", "gay", "gleeful", "gratified", "intoxicated", "jolly", "laughing", "light", "mirthful", "peppy", "perky", "playful", "sparkling", "sunny", "tickled", "tickled pink", "up", "satisfy"]
    temphappy = ' '.join(temphappy)
    tempmotivation = ["catalyst", "desire", "encouragement", "impetus", "impulse", "incentive", "inclination", "interest", "motive", "reason", "wish", "action", "actuation", "angle", "disposition", "drive", "fire", "gimmick",
                      "goose", "hunger", "impulsion", "incitation", "incitement", "inducement", "instigation", "kick", "persuasion", "predetermination", "predisposition", "provocation", "push", "spur", "stimulus", "suggestion", "get", "right stuff"]
    tempmotivation = ' '.join(tempmotivation)
    tempsad = ["sad", "irritate", "lousy", "upset", "incapable", "enraged", "disappointment", "doubtful", "alone", "hostile", "discourage", "uncertain", "paralyze", "insult", "ashame", "indecisive", "fatigue", "sore", "powerless", "perplex", "useless", "annoy", "diminish", "embarrass", "inferior", "upset", "guilty", "hesitant", "vulnerable", "hateful", "dissatisfy", "shy", "empty", "unpleasant", "miserable", "stupefied", "forced", "offensive", "detestable", "disillusion", "hesitant", "bitter", "repugnant", "unbelieving", "despair", "aggressive", "despicable", "skeptical", "frustrated", "resentful", "disgusting", "distrustful", "distress", "inflame", "abominable", "misgiving", "woeful", "provoke", "terrible", "lost", "pathetic", "incensed",
               "indespair", "unsure", "tragic", "infuriate", "sulky", "uneasy", "cross", "bad", "dominate", "tense", "boil", "insensitive", "fearful", "crush", "tearful", "dull", "terrified", "torment", "sorrowful", "nonchalant", "suspicious", "deprive", "neutral", "anxious", "pain", "grief", "reserve", "alarm", "torture", "anguish", "weary", "panic", "deject", "desolate", "bore", "nervous", "reject", "desperate", "preoccupied", "scare", "injure", "pessimistic", "cold", "worry", "offend", "unhappy", "disinterest", "frighten", "afflict", "lonely", "lifeless", "timid", "ache", "grieve", "shaky", "victim", "mourn", "restless", "heartbroken", "dismay", "doubt", "agony", "threaten", "coward", "humiliate", "wrong", "menace", "alienate", "wary"]
    tempsad = ' '.join(tempsad)

    tempvech = nlp(temphappy)
    tempvecl = nlp(templove)
    tempvecm = nlp(tempmotivation)
    tempvecs = nlp(tempsad)

    docvectorh = np.array((tempvech.vector))
    docvectorh = docvectorh.reshape(1, -1)

    docvectorl = np.array((tempvecl.vector))
    docvectorl = docvectorl.reshape(1, -1)

    docvectorm = np.array((tempvecm.vector))
    docvectorm = docvectorm.reshape(1, -1)

    docvectors = np.array((tempvecs.vector))
    docvectors = docvectors.reshape(1, -1)

    arrh = []
    arrl = []
    arrm = []
    arrs = []

    number_of_elements = 10

    if query in temphappy:
        coshappy = (cosine_similarity(docvectorh, vectors)).flatten()
        chappy = coshappy.tolist()
        sortedarray1 = heapq.nlargest(number_of_elements, chappy)
        for x1 in sortedarray1:
            array = chappy.index(x1)
            arrh.append(array)
        for in_ele in arrh:
            dat.append([songnamelist[in_ele], liststr[in_ele]])
        return dat

    elif query in templove:
        coslove = (cosine_similarity(docvectorl, vectors)).flatten()
        clove = coslove.tolist()
        sortedarray2 = heapq.nlargest(number_of_elements, clove)
        for x2 in sortedarray2:
            array = clove.index(x2)
            arrl.append(array)
        for in_ele in arrl:
            dat.append([songnamelist[in_ele], liststr[in_ele]])
        return dat

    elif query in tempmotivation:
        cosmotivation = (cosine_similarity(docvectorm, vectors)).flatten()
        cmotivation = cosmotivation.tolist()
        sortedarray3 = heapq.nlargest(number_of_elements, cmotivation)
        for x3 in sortedarray3:
            array = cmotivation.index(x3)
            arrm.append(array)
        for in_ele in arrm:
            dat.append([songnamelist[in_ele], liststr[in_ele]])
        return dat

    elif query in tempsad:
        cossad = (cosine_similarity(docvectors, vectors)).flatten()
        csad = cossad.tolist()
        sortedarray4 = heapq.nlargest(number_of_elements, csad)
        for x4 in sortedarray4:
            array = csad.index(x4)
            arrs.append(array)
        for in_ele in arrs:
            dat.append([songnamelist[in_ele], liststr[in_ele]])
        return dat

    else:
        pass


@app.route('/')
def home():
    return render_template("search.html")


@app.route('/results', methods=['GET', 'POST'])
def search_request():
    search_term = request.form["keyword"]

    dat.clear()
    dum = backend(search_term)
    pd.set_option('display.max_colwidth', -1)
    df = pd.DataFrame(dum, columns=['Song Name', 'Lyrics'])
    df.style.set_properties(subset=['Song Name'], **{'width': '180px'})
    df.style.set_properties(subset=['Lyrics'], **{'width': '420px'})
    return render_template("results.html", keyword="".join(search_term), data=df.to_html(col_space=50))


if __name__ == "__main__":
    with open('14.csv') as f:
        reader = csv.DictReader(f)

    app.run(debug=True)
