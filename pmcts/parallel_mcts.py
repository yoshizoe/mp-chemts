import cProfile
from math import *
import time
import random
import numpy as np
import random as pr
from copy import deepcopy
import itertools
import sys
from mpi4py import MPI
from threading import Thread, Lock, RLock
from queue import *
from collections import deque
from random import randint
from pmcts.check_ucbpath import backtrack_tdsdfuct, backtrack_mpmcts, compare_ucb_tdsdfuct, compare_ucb_mpmcts, update_selection_ucbtable_mpmcts, update_selection_ucbtable_tdsdfuct
from pmcts.search_tree import Tree_Node
from pmcts.zobrist_hash import Item, HashTable
from enum import Enum

"""
classes defined distributed parallel mcts
"""


class JobType(Enum):
    '''
    defines JobType tag values
    values higher than PRIORITY_BORDER (128) mean high prority tags
    FINISH is not used in this implementation. It will be needed for games.
    '''
    SEARCH = 0
    BACKPROPAGATION = 1
    PRIORITY_BORDER = 128
    TIMEUP = 254
    FINISH = 255

    @classmethod
    def is_high_priority(self, tag):
        return tag >= self.PRIORITY_BORDER.value


class p_mcts:
    """
    parallel mcts algorithms includes TDS-UCT, TDS-df-UCT and MP-MCTS
    """

    def TDS_UCT(chem_model, hsm, property, comm):
        # comm.barrier()
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        status = MPI.Status()

        gau_id = 0  # this is used for wavelength
        allscore = []
        allmol = []
        start_time = time.time()
        _, rootdest = hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if rank == 0:
                if time.time()-start_time > 600:
                    timeup = True
                    for dest in range(1, nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        comm.bsend(dummy_data, dest=dest,
                                   tag=JobType.TIMEUP.value)
            while True:
                ret = comm.Iprobe(source=MPI.ANY_SOURCE,
                                  tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        # high priority messages (timeup and finish)
                        jobq.append(job)
                    else:
                        # normal messages (search and backpropagate)
                        jobq.appendleft(job)

            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    # if node is not in the hash table
                    if hsm.search_table(message[0]) is None:
                        node = Tree_Node(state=message[0], property=property)
                        #node.state = message[0]
                        if node.state == ['&']:
                            node.expansion(chem_model)
                            m = random.choice(node.expanded_nodes)
                            n = node.addnode(m)
                            hsm.insert(Item(node.state, node))
                            _, dest = hsm.hashing(n.state)
                            comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                   n.num_thread_visited]), dest=dest, tag=JobType.SEARCH.value)
                        else:
                            # or max_len_wavelength :
                            if len(node.state) < node.max_len:
                                score, mol = node.simulation(
                                    chem_model, node.state, rank, gau_id)
                                gau_id += 1
                                allscore.append(score)
                                allmol.append(mol)
                                # backpropagation on local memory
                                node.update_local_node(score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state,
                                                       node.reward,
                                                       node.wins,
                                                       node.visits,
                                                       node.num_thread_visited]), dest=dest, tag=JobType.BACKPROPAGATION.value)
                            else:
                                score = -1
                                # backpropagation on local memory
                                node.update_local_node(node, score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state,
                                                       node.reward,
                                                       node.wins,
                                                       node.visits,
                                                       node.num_thread_visited]),
                                           dest=dest, tag=JobType.BACKPROPAGATION.value)

                    else:  # if node already in the local hashtable
                        node = hsm.search_table(message[0])
                        print("debug:", node.visits,
                              node.num_thread_visited, node.wins)
                        if node.state == ['&']:
                            if node.expanded_nodes != []:
                                m = random.choice(node.expanded_nodes)
                                n = node.addnode(m)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(n.state)
                                comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                       n.num_thread_visited]), dest=dest, tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(childnode.state)
                                comm.bsend(np.asarray([childnode.state,
                                                       childnode.reward,
                                                       childnode.wins,
                                                       childnode.visits,
                                                       childnode.num_thread_visited]),
                                           dest=dest,
                                           tag=JobType.SEARCH.value)
                        else:
                            #node.num_thread_visited = message[4]
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = random.choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        hsm.insert(Item(node.state, node))
                                        _, dest = hsm.hashing(n.state)
                                        comm.bsend(np.asarray([n.state,
                                                               n.reward, n.wins, n.visits,
                                                               n.num_thread_visited]),
                                                   dest=dest, tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(chem_model)
                                            m = random.choice(
                                                node.expanded_nodes)
                                            n = node.addnode(m)
                                            hsm.insert(Item(node.state, node))
                                            _, dest = hsm.hashing(n.state)
                                            comm.bsend(np.asarray([n.state,
                                                                   n.reward, n.wins, n.visits,
                                                                   n.num_thread_visited]),
                                                       dest=dest, tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            hsm.insert(Item(node.state, node))
                                            _, dest = hsm.hashing(
                                                childnode.state)
                                            comm.bsend(np.asarray([childnode.state,
                                                                   childnode.reward,
                                                                   childnode.wins,
                                                                   childnode.visits,
                                                                   childnode.num_thread_visited]),
                                                       dest=dest, tag=JobType.SEARCH.value)

                                else:
                                    score, mol = node.simulation(
                                        chem_model, node.state, rank, gau_id)

                                    gau_id += 1
                                    score = -1
                                    allscore.append(score)
                                    allmol.append(mol)
                                    # backpropagation on local memory
                                    node.update_local_node(score)
                                    hsm.insert(Item(node.state, node))
                                    _, dest = hsm.hashing(node.state[0:-1])
                                    comm.bsend(np.asarray([node.state,
                                                           node.reward,
                                                           node.wins,
                                                           node.visits,
                                                           node.num_thread_visited]),
                                               dest=dest, tag=JobType.BACKPROPAGATION.value)

                            else:
                                score = -1
                                # backpropagation on local memory
                                node.update_local_node(score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state,
                                                       node.reward,
                                                       node.wins,
                                                       node.visits,
                                                       node.num_thread_visited]),
                                           dest=dest,
                                           tag=JobType.BACKPROPAGATION.value)

                elif tag == JobType.BACKPROPAGATION.value:
                    node = Tree_Node(state=message[0], property=property)
                    node.reward = message[1]
                    local_node = hsm.search_table(message[0][0:-1])
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        hsm.insert(Item(local_node.state, local_node))
                        _, dest = hsm.hashing(local_node.state)
                        comm.bsend(np.asarray([local_node.state,
                                               local_node.reward,
                                               local_node.wins,
                                               local_node.visits,
                                               local_node.num_thread_visited]),
                                   dest=dest,
                                   tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        hsm.insert(Item(local_node.state, local_node))
                        _, dest = hsm.hashing(local_node.state[0:-1])
                        comm.bsend(np.asarray([local_node.state,
                                               local_node.reward,
                                               local_node.wins,
                                               local_node.visits,
                                               local_node.num_thread_visited]),
                                   dest=dest,
                                   tag=JobType.BACKPROPAGATION.value)
                elif tag == JobType.TIMEUP.value:
                    timeup = True

        return allscore, allmol

    def TDS_df_UCT(chem_model, hsm, property, comm):
        # comm.barrier()
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        status = MPI.Status()
        gau_id = 0  # this is used for wavelength
        start_time = time.time()
        allscore = []
        allmol = []
        depth = []
        bpm = 0
        bp = []
        _, rootdest = hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if rank == 0:
                if time.time()-start_time > 600:
                    timeup = True
                    for dest in range(1, nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        comm.bsend(dummy_data, dest=dest,
                                   tag=JobType.TIMEUP.value)
            while True:
                ret = comm.Iprobe(source=MPI.ANY_SOURCE,
                                  tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = comm.recv(
                        source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        jobq.append(job)
                    else:
                        jobq.appendleft(job)
            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    if hsm.search_table(message[0]) == None:
                        node = Tree_Node(state=message[0], property=property)
                        info_table = message[5]
                        #print ("not in table info_table:",info_table)
                        if node.state == ['&']:
                            node.expansion(chem_model)
                            m = random.choice(node.expanded_nodes)
                            n = node.addnode(m)
                            hsm.insert(Item(node.state, node))
                            _, dest = hsm.hashing(n.state)
                            comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                   n.num_thread_visited, info_table]),
                                       dest=dest,
                                       tag=JobType.SEARCH.value)
                        else:
                            if len(node.state) < node.max_len:
                                score, mol = node.simulation(
                                    chem_model, node.state, rank, gau_id)
                                #print (mol)
                                gau_id += 1
                                allscore.append(score)
                                allmol.append(mol)
                                depth.append(len(node.state))
                                node.update_local_node(score)
                                # update infor table
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                       node.num_thread_visited, info_table]),
                                           dest=dest,
                                           tag=JobType.BACKPROPAGATION.value)
                            else:
                                score = -1
                                node.update_local_node(node, score)
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                       node.num_thread_visited, info_table]),
                                           dest=dest,
                                           tag=JobType.BACKPROPAGATION.value)

                    else:  # if node already in the local hashtable
                        node = hsm.search_table(message[0])
                        info_table = message[5]
                        #print ("in table info_table:",info_table)
                        if node.state == ['&']:
                            # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                            if node.expanded_nodes != []:
                                m = random.choice(node.expanded_nodes)
                                n = node.addnode(m)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(n.state)
                                comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                       n.num_thread_visited, info_table]),
                                           dest=dest,
                                           tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                hsm.insert(Item(node.state, node))
                                info_table = update_selection_ucbtable_tdsdfuct(
                                    info_table, node, ind)
                                #print ("info_table after selection:",info_table)
                                _, dest = hsm.hashing(childnode.state)
                                comm.bsend(np.asarray([childnode.state, childnode.reward, childnode.wins,
                                                       childnode.visits, childnode.num_thread_visited,
                                                       info_table]),
                                           dest=dest, tag=JobType.SEARCH.value)
                        else:
                            #node.path_ucb = message[5]
                            # info_table=message[5]
                            #print("check ucb:", node.reward, node.visits, node.num_thread_visited,info_table)
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = random.choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        hsm.insert(Item(node.state, node))
                                        _, dest = hsm.hashing(n.state)
                                        comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                               n.num_thread_visited, info_table]),
                                                   dest=dest,
                                                   tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(chem_model)
                                            m = random.choice(
                                                node.expanded_nodes)
                                            n = node.addnode(m)
                                            hsm.insert(Item(node.state, node))
                                            _, dest = hsm.hashing(n.state)
                                            comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                                   n.num_thread_visited, info_table]),
                                                       dest=dest,
                                                       tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            hsm.insert(Item(node.state, node))
                                            info_table = update_selection_ucbtable_tdsdfuct(
                                                info_table, node, ind)
                                            _, dest = hsm.hashing(
                                                childnode.state)
                                            comm.bsend(np.asarray([childnode.state, childnode.reward, childnode.wins,
                                                                   childnode.visits, childnode.num_thread_visited, info_table]),
                                                       dest=dest,
                                                       tag=JobType.SEARCH.value)
                                else:
                                    score, mol = node.simulation(
                                        chem_model, node.state, rank, gau_id)
                                    gau_id += 1
                                    score = -1
                                    allscore.append(score)
                                    allmol.append(mol)
                                    depth.append(len(node.state))
                                    node.update_local_node(score)
                                    info_table = backtrack_tdsdfuct(
                                        info_table, score)

                                    hsm.insert(Item(node.state, node))
                                    _, dest = hsm.hashing(node.state[0:-1])
                                    comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                           node.num_thread_visited, info_table]),
                                               dest=dest,
                                               tag=JobType.BACKPROPAGATION.value)
                            else:
                                score = -1
                                node.update_local_node(score)
                                info_table = backtrack_tdsdfuct(
                                    info_table, score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state, node.reward, node.wins,
                                                       node.visits, node.num_thread_visited, info_table]),
                                           dest=dest,
                                           tag=JobType.BACKPROPAGATION.value)

                elif tag == JobType.BACKPROPAGATION.value:
                    bpm += 1
                    node = Tree_Node(state=message[0], property=property)
                    node.reward = message[1]
                    local_node = hsm.search_table(message[0][0:-1])
                    #print ("report check message[5]:",message[5])
                    #print ("check:",len(message[0]), len(message[5]))
                    #print ("check:",local_node.wins, local_node.visits, local_node.num_thread_visited)
                    info_table=message[5]
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        hsm.insert(Item(local_node.state, local_node))
                        _, dest = hsm.hashing(local_node.state)
                        comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                               local_node.visits, local_node.num_thread_visited, info_table]),
                                               dest=dest,
                                               tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        #local_node,info_table = backtrack_tdsdf(info_table,local_node, node)
                        back_flag = compare_ucb_tdsdfuct(info_table,local_node)
                        hsm.insert(Item(local_node.state, local_node))
                        if back_flag == 1:
                            _, dest = hsm.hashing(local_node.state[0:-1])
                            comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                local_node.visits, local_node.num_thread_visited, info_table[0:-1]]),
                                                   dest=dest,
                                                   tag=JobType.BACKPROPAGATION.value)
                        if back_flag == 0:
                            _, dest = hsm.hashing(local_node.state)
                            comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                                   local_node.visits, local_node.num_thread_visited, info_table]),
                                                   dest=dest,
                                                   tag=JobType.SEARCH.value)
                elif tag == JobType.TIMEUP.value:
                    timeup = True
        bp.append(bpm)

        return allscore, allmol

    def MP_MCTS(chem_model, hsm, property, comm):
        #comm.barrier()
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        status = MPI.Status()
        gau_id = 0 ## this is used for wavelength
        start_time = time.time()
        allscore = []
        allmol = []
        _, rootdest = hsm.hashing(['&'])
        jobq = deque()
        timeup = False
        if rank == rootdest:
            root_job_message = np.asarray([['&'], None, 0, 0, 0, []])
            for i in range(3 * nprocs):
                temp = deepcopy(root_job_message)
                root_job = (JobType.SEARCH.value, temp)
                jobq.appendleft(root_job)
        while not timeup:
            if rank == 0:
                if time.time()-start_time > 60:
                    timeup = True
                    for dest in range(1, nprocs):
                        dummy_data = tag = JobType.TIMEUP.value
                        comm.bsend(dummy_data, dest=dest, tag=JobType.TIMEUP.value)
            while True:
                ret = comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                if ret == False:
                    break
                else:
                    message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    cur_status = status
                    tag = cur_status.Get_tag()
                    job = (tag, message)
                    if JobType.is_high_priority(tag):
                        jobq.append(job)
                    else:
                        jobq.appendleft(job)
            jobq_non_empty = bool(jobq)
            if jobq_non_empty:
                (tag, message) = jobq.pop()
                if tag == JobType.SEARCH.value:
                    if hsm.search_table(message[0]) == None:
                        node = Tree_Node(state=message[0], property=property)
                        if node.state == ['&']:
                            node.expansion(chem_model)
                            m = random.choice(node.expanded_nodes)
                            n = node.addnode(m)
                            hsm.insert(Item(node.state, node))
                            _, dest = hsm.hashing(n.state)
                            comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                   n.num_thread_visited, n.path_ucb]),
                                                   dest=dest,
                                                   tag=JobType.SEARCH.value)
                        else:
                            if len(node.state) < node.max_len:
                                score, mol = node.simulation(chem_model, node.state, rank, gau_id)
                                gau_id+=1
                                allscore.append(score)
                                allmol.append(mol)
                                node.update_local_node(score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                       node.num_thread_visited, node.path_ucb]),
                                                       dest=dest,
                                                       tag=JobType.BACKPROPAGATION.value)
                            else:
                                score = -1
                                node.update_local_node(node, score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                       node.num_thread_visited, node.path_ucb]),
                                                       dest=dest,
                                                       tag=JobType.BACKPROPAGATION.value)

                    else:  # if node already in the local hashtable
                        node = hsm.search_table(message[0])
                        if node.state == ['&']:
                            # print ("in table root:",node.state,node.path_ucb,len(node.state),len(node.path_ucb))
                            if node.expanded_nodes != []:
                                m = random.choice(node.expanded_nodes)
                                n = node.addnode(m)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(n.state)
                                comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                       n.num_thread_visited, n.path_ucb]),
                                                       dest=dest,
                                                       tag=JobType.SEARCH.value)
                            else:
                                ind, childnode = node.selection()
                                hsm.insert(Item(node.state, node))
                                ucb_table = update_selection_ucbtable_mpmcts(node, ind)
                                _, dest = hsm.hashing(childnode.state)
                                comm.bsend(np.asarray([childnode.state, childnode.reward, childnode.wins,
                                                       childnode.visits, childnode.num_thread_visited,
                                                       ucb_table]),
                                                       dest=dest,tag=JobType.SEARCH.value)
                        else:
                            node.path_ucb = message[5]
                            print("check ucb:", node.wins, node.visits, node.num_thread_visited)
                            if len(node.state) < node.max_len:
                                if node.state[-1] != '\n':
                                    if node.expanded_nodes != []:
                                        m = random.choice(node.expanded_nodes)
                                        n = node.addnode(m)
                                        hsm.insert(Item(node.state, node))
                                        _, dest = hsm.hashing(n.state)
                                        comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                               n.num_thread_visited, n.path_ucb]),
                                                               dest=dest,
                                                               tag=JobType.SEARCH.value)
                                    else:
                                        if node.check_childnode == []:
                                            node.expansion(chem_model)
                                            m = random.choice(node.expanded_nodes)
                                            n = node.addnode(m)
                                            hsm.insert(Item(node.state, node))
                                            _, dest = hsm.hashing(n.state)
                                            comm.bsend(np.asarray([n.state, n.reward, n.wins, n.visits,
                                                                   n.num_thread_visited, n.path_ucb]),
                                                                   dest=dest,
                                                                   tag=JobType.SEARCH.value)
                                        else:
                                            ind, childnode = node.selection()
                                            hsm.insert(Item(node.state, node))
                                            ucb_table = update_selection_ucbtable_mpmcts(node, ind)
                                            _, dest = hsm.hashing(childnode.state)
                                            comm.bsend(np.asarray([childnode.state, childnode.reward, childnode.wins,
                                                                   childnode.visits, childnode.num_thread_visited, ucb_table]),
                                                                   dest=dest,
                                                                   tag=JobType.SEARCH.value)
                                else:
                                    score, mol = node.simulation(chem_model, node.state, rank, gau_id)
                                    gau_id+=1
                                    score = -1
                                    allscore.append(score)
                                    allmol.append(mol)
                                    node.update_local_node(score)
                                    hsm.insert(Item(node.state, node))
                                    _, dest = hsm.hashing(node.state[0:-1])
                                    comm.bsend(np.asarray([node.state, node.reward, node.wins, node.visits,
                                                           node.num_thread_visited, node.path_ucb]),
                                                           dest=dest,
                                                           tag=JobType.BACKPROPAGATION.value)
                            else:
                                score = -1
                                node.update_local_node(score)
                                hsm.insert(Item(node.state, node))
                                _, dest = hsm.hashing(node.state[0:-1])
                                comm.bsend(np.asarray([node.state, node.reward, node.wins,
                                                       node.visits, node.num_thread_visited, node.path_ucb]),
                                                       dest=dest,
                                                       tag=JobType.BACKPROPAGATION.value)

                elif tag == JobType.BACKPROPAGATION.value:
                    node = Tree_Node(state=message[0], property=property)
                    node.reward = message[1]
                    local_node = hsm.search_table(message[0][0:-1])
                    if local_node.state == ['&']:
                        local_node.backpropagation(node)
                        hsm.insert(Item(local_node.state, local_node))
                        _, dest = hsm.hashing(local_node.state)
                        comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                               local_node.visits, local_node.num_thread_visited, local_node.path_ucb]),
                                               dest=dest,
                                               tag=JobType.SEARCH.value)
                    else:
                        local_node.backpropagation(node)
                        local_node = backtrack_mpmcts(local_node, node)
                        back_flag = compare_ucb_mpmcts(local_node)
                        hsm.insert(Item(local_node.state, local_node))
                        if back_flag == 1:
                            _, dest = hsm.hashing(local_node.state[0:-1])
                            comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                                   local_node.visits, local_node.num_thread_visited, local_node.path_ucb]),
                                                   dest=dest,
                                                   tag=JobType.BACKPROPAGATION.value)
                        if back_flag == 0:
                            _, dest = hsm.hashing(local_node.state)
                            comm.bsend(np.asarray([local_node.state, local_node.reward, local_node.wins,
                                                   local_node.visits, local_node.num_thread_visited, local_node.path_ucb]),
                                                   dest=dest,
                                                   tag=JobType.SEARCH.value)
                elif tag == JobType.TIMEUP.value:
                    timeup = True

        return allscore, allmol
