import random
import numpy as np
from mpi4py import MPI
import csv
from pmcts.load_model import loaded_logp_model, loaded_wave_model
from pmcts.zobrist_hash import Item, HashTable
from pmcts.search_tree import Tree_Node
from pmcts.write_to_csv import wcsv
from pmcts.parallel_mcts import p_mcts


if __name__ == "__main__":
    """
    Initialize MPI environment
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    status = MPI.Status()
    mem = np.zeros(1024 * 10 * 1024)
    MPI.Attach_buffer(mem)

    """
    Load the pre-trained RNN model and define the property optimized:
    currently available properties: logP (rdkit) and wavelength (DFT)
    """
    chem_model = loaded_logp_model()
    property = "logP"
    node = Tree_Node(state=['&'], property=property)

    """
    Initialize HashTable
    """
    random.seed(3)
    hsm = HashTable(nprocs, node.val, node.max_len, len(node.val))

    """
    Design molecules using parallel MCTS: TDS-UCT,TDS-df-UCT and MP-MCTS
    """
    comm.barrier()
    #score,mol=p_mcts.TDS_UCT(chem_model, hsm, property, comm)
    #score,mol=p_mcts.TDS_df_UCT(chem_model, hsm, property, comm)
    score, mol = p_mcts.MP_MCTS(chem_model, hsm, property, comm)

    wcsv(score, 'logp_score' + str(rank))
    wcsv(mol, 'logp_mol' + str(rank))
    # wcsv(score, 'logp_dmcts_scoreForProcess' + str(rank))
    # wcsv(mol, 'logp_dmcts_generatedMoleculesForProcess' + str(rank))
    #wcsv(depth,'depth' + str(rank))
