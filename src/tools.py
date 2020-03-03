
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
    '''
    function whcih shows how far you are through a loop
    
    Attributes
    progress (float):  index of(current step) / len(iterable)
    
    Returns: 
    a progress bar
    '''
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
            color = 'green' if cg.weight > 0 else 'blue'
            width = str(0.1 + abs(cg.weight / 2.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot



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
    out_lst = [precision, recall]
    return out_lst