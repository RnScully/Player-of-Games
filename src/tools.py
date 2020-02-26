
import time, sys
from IPython.display import clear_output
import pickle
import neat
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import graphviz

from sklearn.metrics import confusion_matrix

def update_progress(progress):
    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def save_model(model,name):
    file_ext= '.sav'
    path = 'models/'
    pickle.dump(model, open(path+name+file_ext, 'wb'))

def extract_ngrams(data, num):
    n_grams = ngrams(nltk.word_tokenize(data), num)
    return [ ' '.join(grams) for grams in n_grams]

def draw_net(config_path, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


def vectorizable_ngrams(original_words_list):

    my_stop_words = set([',', # punctuation
                    '.',
                    '"',
                    "'",
                    'I',# capital I keeps getting into the ngrams? 
                    'a',# the below are stopwords from nltk with all of the negation terms removed
                    'about',
                    'across',
                    'after',
                    'afterwards',
                    'again','all','almost','alone','along',
                    'already',
                    'also',
                    'although',
                    'always',
                    'am',
                    'among',
                    'amongst',
                    'amoungst',
                    'amount',
                    'an',
                    'and',
                    'another',
                    'any',
                    'anyhow',
                    'anyone',
                    'anything',
                    'anyway',
                    'anywhere',
                    'are',
                    'around',
                    'as',
                    'at',
                    'back',
                    'be',
                    'became',
                    'because',
                    'become',
                    'becomes',
                    'becoming',
                    'been',
                    'before',
                    'beforehand',
                    'being',
                    'beside',
                    'besides',
                    'between',
                    'beyond',
                    'bill',
                    'both',
                    'bottom',
                    'but',
                    'by',
                    'call',
                    'de',
                    'describe',
                    'detail',
                    'do',
                    'done',
                    'down',
                    'due',
                    'during',
                    'each',
                    'eg',
                    'eight',
                    'either',
                    'eleven',
                    'else',
                    'elsewhere',
                    'empty',
                    'enough',
                    'etc',
                    'even',
                    'ever',
                    'every',
                    'everyone',
                    'everything',
                    'everywhere',
                    'few',
                    'fifteen',
                    'fifty',
                    'fill',
                    'find',
                    'fire',
                    'first',
                    'five',
                    'for',
                    'former',
                    'formerly',
                    'forty',
                    'found',
                    'four',
                    'from',
                    'front',
                    'full',
                    'further',
                    'get',
                    'give',
                    'go',
                    'had',
                    'has',
                    'have',
                    'he',
                    'hence',
                    'her',
                    'here',
                    'hereafter',
                    'hereby',
                    'herein',
                    'hereupon',
                    'hers',
                    'herself',
                    'him',
                    'himself',
                    'his',
                    'how',
                    'however',
                    'hundred',
                    'i',
                    'ie',
                    'if',
                    'in',
                    'inc',
                    'indeed',
                    'into',
                    'is',
                    'it',
                    'its',
                    'itself',
                    'keep',
                    'last',
                    'latter',
                    'latterly',
                    'least',
                    'less',
                    'ltd',
                    'made',
                    'many',
                    'may',
                    'me',
                    'meanwhile',
                    'might',
                    'mill',
                    'mine',
                    'more',
                    'moreover',
                    'most',
                    'mostly',
                    'move',
                    'much',
                    'must',
                    'my',
                    'myself',
                    'name',
                    'namely',
                    'next',
                    'nine',
                    'of',
                    'off',
                    'often',
                    'on',
                    'once',
                    'one',
                    'only',
                    'onto',
                    'or',
                    'other',
                    'others',
                    'otherwise',
                    'our',
                    'ours',
                    'ourselves',
                    'out',
                    'over',
                    'own',
                    'part',
                    'per',
                    'perhaps',
                    'please',
                    'put',
                    'rather',
                    're',
                    'same',
                    'see',
                    'seem',
                    'seemed',
                    'seeming',
                    'seems',
                    'serious',
                    'several',
                    'she',
                    'should',
                    'show',
                    'side',
                    'since',
                    'sincere',
                    'six',
                    'sixty',
                    'so',
                    'some',
                    'somehow',
                    'someone',
                    'something',
                    'sometime',
                    'sometimes',
                    'somewhere',
                    'still',
                    'such',
                    'system',
                    'take',
                    'ten',
                    'than',
                    'that',
                    'the',
                    'their',
                    'them',
                    'themselves',
                    'then',
                    'thence',
                    'there',
                    'thereafter',
                    'thereby',
                    'therefore',
                    'therein',
                    'thereupon',
                    'these',
                    'they',
                    'thick',
                    'thin',
                    'third',
                    'this',
                    'those',
                    'though',
                    'three',
                    'through',
                    'throughout',
                    'thru',
                    'thus',
                    'to',
                    'together',
                    'too',
                    'top',
                    'toward',
                    'towards',
                    'twelve',
                    'twenty',
                    'two',
                    'un',
                    'under',
                    'until',
                    'up',
                    'upon',
                    'us',
                    'via',
                    'was',
                    'we',
                    'well',
                    'were',
                    'what',
                    'whatever',
                    'when',
                    'whence',
                    'whenever',
                    'where',
                    'whereafter',
                    'whereas',
                    'whereby',
                    'wherein',
                    'whereupon',
                    'wherever',
                    'whether',
                    'which',
                    'while',
                    'whither',
                    'who',
                    'whoever',
                    'whole',
                    'whom',
                    'whose',
                    'why',
                    'will',
                    'with',
                    'within',
                    'would',
                    'yet',
                    'you',
                    'your',
                    'yours',
                    'yourself',
                    'yourselves'
                    ])
                        

    periods = [i.replace('.','. ') for i in original_words_list]
    data =[]
    for rows in periods:
        word_tokens = word_tokenize(rows) 
        data.append(' '.join([w.lower() for w in word_tokens if not w.lower() in my_stop_words]))
    ngram_lst =[extract_ngrams(i,2) for i in data]
    dashngrams = [[grams.replace(' ','-') for grams in ngrams] for ngrams in ngram_lst]
    return dashngrams

def predict_one(review):
    '''
    Parameters
    review (str): a text review
    '''
    path = 'models/'
    
    tfid = pickle.load(open(path+'tf84.sav', 'rb'))
    model = pickle.load(open(path+'ngramRF85.sav', 'rb'))
    
    ngrams = vectorizable_ngrams([review]) #string must becalled as a list to handle single strin
    
    tfstring = ' '.join(ngrams[0])
    
    print(tfstring) ##what it sees, hash it out for final. 
    tfidfed = tfid.transform([tfstring])
    
    return model.predict(tfidfed)

def predict_many(review_list):
    path = 'models/'

    tfid = pickle.load(open(path+'tf84.sav', 'rb'))
    model = pickle.load(open(path+'ngramRF85.sav', 'rb'))

    ngrams = vectorizable_ngrams(reviews) #string must becalled as a list to handle single strin
    tfstring = [' '.join(items) for items in ngrams]

    #print(tfstring) ##what it sees, hash it out for final. 

    tfidfed = tfid.transform(tfstring)

    return model.predict(tfidfed)


# from sklearn.metrics import confusion_matrix
#

def report_score(model, X_train, X_test, y_train, y_test):
    
    '''
    a function that reports the accuracy of the model.
    Attributes:
    models (lst): a list of instansiated models to test
    Returns:
    out array, model name, training score, testing score, precision, recall
    '''
    #training_score = model.score(X_train, y_train) ## Many models don't have scores!
    #testing_score = model.score(X_test, y_test)
    #print('Training score: {}, Testing score: {}'.format(training_score, testing_score))
    tn, fp, fn, tp = confusion_matrix(y_test,model.predict(X_test)).ravel()
    precision = tp/(fp+tp)
    recall = tp/(fn+tp)
    print('tn', '  fp', '  fn', '  tp')
    print(tn, fp, fn, tp)
    print('precision: '+str(precision), 'recall: '+ str(recall))
    out_lst = [training_score, testing_score, precision, recall]
    return out_lst

def remove_parts_of_speech(tokenized_corpus, parts_to_remove):
    '''function which uses nltk position tagging to remove parts of speach
    ++++++++++
    Attributes
    tokenized_corpus (lst): a list of lists: the corpus of documents, each doc transformed into a list of tokens
    parts_to_remove (lst): a list of the NLTK parts of speach that you want removed

    Returns: 
    ++++++++++
    no_pronouns(lst): a list of lists simmilar to tokenized_corpus containing none of the parts of speech you wanted removed
    '''
    remove = set(parts_to_remove)
    no_pronouns = []
    for text in tokenized_corpus:
        j =nltk.pos_tag(text)
        review = []
        for pos in j:
            if pos[1] in remove:
                continue
            else:
                review.append(pos[0])
        no_pronouns.append(review)

    return no_pronouns