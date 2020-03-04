
import time, sys
from IPython.display import clear_output
import pickle
import neat

import graphviz


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
    '''
    given a model object, saves that object via pickle to a models directory
    ++++
    Parameter
    model (object, usually sikit model): object to save
    
    '''
    file_ext= '.sav'
    path = 'models/'
    pickle.dump(model, open(path+name+file_ext, 'wb'))



def draw_net(config_path, genome, view=False, filename=None, node_names=None, show_disabled=False, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ 
    Code pulled from code reclaimers at https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py
    Receives a genome and draws a neural network with arbitrary topology. 
    Parameters
    config_path (str): path to the neat configuration file used to evolve this net
    genome (neat object): genome of nn you are drawing
    view (bool): True will show it inline, or in your notebook, False will not
    filename (str): what to name the diagram that this saves. 
    node_names I don't know what this does
    show_disabled (bool) False will show links in nodes that this induvidual has deactivated. # I really believe this makes no sense...the ai does not have that part if the genome has deactivated it.  
    prune_unused (bool) :not sure what this one does, it might have something to do with how show_disabled doesn't make sense. 
    
    """
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







# from sklearn.metrics import confusion_matrix
#

