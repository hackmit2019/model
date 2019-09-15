import spacy
import numpy as np
from numpy.linalg import norm
import math
import nltk
import torch
nltk.download('punkt')

def infer(inputs):
    radius = 0.09

    sentences = []
    locations = []

    import json
    d = json.loads(inputs)

    for call in d:
        sentences.append(call['transcript'])
        locations.append((call['latitude'], call['longitude']))

    from models import InferSent
    V = 2
    MODEL_PATH = 'encoder/infersent%s.pkl' % V
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    W2V_PATH = 'fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)

    ## The old, bag of filtered words, implementation follows
    # for i, sentence in enumerate(sentences):
    #     sentences[i] = nlp(' '.join([str(t) for t in nlp(sentence) if t.pos_ in ['NOUN', 'PROPN', 'ADJ']]))
    #
    # sentences_matrix = np.vstack([x.vector / norm(x.vector) for x in sentences])
    # ling_compatibility = np.matmul(sentences_matrix, np.transpose(sentences_matrix))
    # print(ling_compatibility)

    infersent.build_vocab(sentences, tokenize=True)
    embeddings = infersent.encode(sentences, tokenize=True)
    embeddings = embeddings/np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)


    ling_compatibility = np.matmul(embeddings, np.transpose(embeddings))
    #print(ling_compatibility)

    def intersection_area(d, r):
        if d == 0:  # the circles are the same
            return np.pi * r**2
        if d >= 2 * r:  # The circles don't overlap at all.
            return 0

        r2, d2 = r**2, d**2
        alpha = np.arccos(d2 / (2*d*r))
        wow = 2 * r2 * alpha - r2 * np.sin(2*alpha)
        return wow

    geo_compatibility = np.zeros((len(locations), len(locations)))
    for i in range(len(locations)):
        for k in range(i, len(locations)):
            geo_compatibility[i][k] = intersection_area(math.sqrt((locations[i][0] - locations[k][0])**2 + (locations[i][1] - locations[k][1])**2), radius) / (math.pi * (2**2))

    from sklearn.cluster import KMeans
    total = np.multiply(ling_compatibility, geo_compatibility)
    #print(total.shape)
    #for i in range(len(locations)):
    #    for k in range(len(locations)):
    #        if i != k and total[i][k] > 0.65:
    #            print(str(i) + " and " + str(k) + " are the same incident")
    kmeany = KMeans(init='k-means++').fit(total)
    labels = kmeany.labels_.tolist()

    #print(labels)
    mapper = {}
    for call, label in enumerate(labels):
        mapper[call] = label

    return json.dumps(mapper)

print(infer(inputs ='[{"latitude":33.44382517445177,"longitude":-86.80971837149715,"transcript":"Oh no! My house has flooded."},{"latitude":33.44774107052392,"longitude":-86.78090872701958,"transcript":"Oh no, a telephone pole has fallen down."},{"latitude":33.443524674882106,"longitude":-86.81592694945077,"transcript":"Oh no, my neighbor\'s house has flooded!"},{"latitude":33.437449745654455,"longitude":-86.78451802590659,"transcript":"There is water filling the streets!"},{"latitude":33.44676504709364,"longitude":-86.82388588570537,"transcript":"It is raining too hard and won\'t stop. The water level has already reached into the houses of my neighbor."},{"latitude":33.44022014738728,"longitude":-86.79524213982236,"transcript":"There\'s too much rain, and it is causing flooding."},{"latitude":33.44355542779387,"longitude":-86.79276784607535,"transcript":"I saw a telephone poll fall down recently."},{"latitude":33.44902111978129,"longitude":-86.81666408448383,"transcript":"I don\'t know where my dog is. I seem to have lost him."},{"latitude":33.442312404110844,"longitude":-86.8350656002557,"transcript":"Help, my baby drowned in the flooding."},{"latitude":33.432222654798956,"longitude":-86.8166856005463,"transcript":"Help, my dog\'s baby has drowned in the flood."},{"latitude":33.47064069811926,"longitude":-86.94198082118417,"transcript":"There is a tiger loose in the city."},{"latitude":33.47393326021101,"longitude":-86.94025770588739,"transcript":"There is a wild animal who has exited the zoo and is hungry for babies."},{"latitude":33.478148897414684,"longitude":-86.95884286229291,"transcript":"I\'m worried my baby will be eaten by the tiger."},{"latitude":33.488630335901846,"longitude":-86.96745222298738,"transcript":"I let a tiger loose because I don\'t like my neighbor\'s baby."},{"latitude":33.55948920796703,"longitude":-86.87988103189846,"transcript":"I just witnessed a car crash, and I suspect the people involved are drug dealers."},{"latitude":33.5610566742092,"longitude":-86.9042067254693,"transcript":"My dealer just crashed into this asshole, and the meth I just bought is now gone."},{"latitude":33.549666565411464,"longitude":-86.89212923722228,"transcript":"There\'s a car crash on 35 and lamar."},{"latitude":33.55813756321193,"longitude":-86.90198170032507,"transcript":"Help, I just got into an accident with a drug dealer on 35 and lamar."},{"latitude":33.553768209731274,"longitude":-86.90779487406523,"transcript":"I saw a telephone poll fall down. I\'m worried someone may be hurt."},{"latitude":33.57029520464136,"longitude":-86.89535620128386,"transcript":"There was a pretty severe pileup on 35 and lamar."},{"latitude":33.57219163430416,"longitude":-86.94708595687497,"transcript":"The hurricane has begun to hit the city, and I haven\'t evacuated yet."},{"latitude":33.581899810532256,"longitude":-86.92677666376132,"transcript":"Help! My baby is stuck underneath the floorboards and I can\'t go back into the house because of the hurricane."},{"latitude":33.56619672794073,"longitude":-86.91240915549824,"transcript":"I am trying to smoke but the strong winds of the hurricane keep blowing out my joint."},{"latitude":33.57290111789184,"longitude":-86.94491629030934,"transcript":"There is a very big storm with lots of rain and strong winds. I am afraid of being hurled around like Dorothy and the Aluminum man."},{"latitude":33.573750082681066,"longitude":-86.90766806001234,"transcript":"I thought the hurricane would not hit as far inland as Birmingham, but I was wrong! I am currently aloft due to the strong winds and I fear I may not make it back down."},{"latitude":33.58287907350006,"longitude":-86.93978365892613,"transcript":"There is strong winds and a massive downpour. I fear flooding. Please send help to 26th and Speedway."},{"latitude":33.599776052156926,"longitude":-86.92589680041708,"transcript":"Please evacuate the city, because there is a storm inbound that will probably destroy the city."},{"latitude":33.57221562691879,"longitude":-86.9352188997241,"transcript":"There\'s a hurricane! Why has the city not evacuated me yet? I am blown away, and about to be for real!"},{"latitude":33.54726908689364,"longitude":-86.92806674178114,"transcript":"I have made peace with my impending death, but others may not have. Please alert the public of this hurricane."}]'))