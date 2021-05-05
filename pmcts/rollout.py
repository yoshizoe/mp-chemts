from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from keras.preprocessing import sequence
from pmcts import sascorer
import gzip
import networkx as nx
import numpy as np

"""Sampling molecules in simulation step"""
def chem_kn_simulation(model, state, val, max_len):
    all_posible = []
    end = "\n"
    position = []
    position.extend(state)
    total_generated = []
    new_compound = []
    get_int_old = []
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))
    get_int = get_int_old
    x = np.reshape(get_int, (1, len(get_int)))
    x_pad = sequence.pad_sequences(x, maxlen=max_len, dtype='int32',
                                   padding='post', truncating='pre', value=0.)
    while not get_int[-1] == val.index(end):
        predictions = model.predict(x_pad)
        preds = np.asarray(predictions[0][len(get_int) - 1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int = np.argmax(next_probas)
        get_int.append(next_int)
        x = np.reshape(get_int, (1, len(get_int)))
        x_pad = sequence.pad_sequences(x, maxlen=max_len, dtype='int32', padding='post',
                                    truncating='pre', value=0.)
        if len(get_int) > max_len:
            break
    total_generated.append(get_int)
    all_posible.extend(total_generated)
    return all_posible


def predict_smile(all_posible, val):
    new_compound = []
    for i in range(len(all_posible)):
        total_generated = all_posible[i]
        generate_smile = []
        for j in range(len(total_generated) - 1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)
    return new_compound


def make_input_smile(generate_smile):
    new_compound = []
    for i in range(len(generate_smile)):
        middle = []
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com = ''.join(middle)
        new_compound.append(com)
    return new_compound
