from random import randint
import math


class Item:
    key = ""
    value = 0

    def __init__(self, key, value):
        self.key = key
        self.value = value


class HashTable:
    'Common base class for a hash table'
    tableSize = 0
    entriesCount = 0
    alphabetSize = 2 * 26
    hashTable = []

    def __init__(self, nprocs, val, max_len, val_len):
        self.hashTable = dict()  # [[] for i in range(size)]

        # should be enough for our case, but should be larger for longer search
        self.hash_index_bits = 32
        self.hash_table_max_size = 2**self.hash_index_bits

        self.S = max_len
        self.P = val_len
        self.val = val
        self.nprocs = nprocs
        self.zobristnum = [[0] * self.P for i in range(self.S)]
        for i in range(self.S):
            for j in range(self.P):
                self.zobristnum[i][j] = randint(0, 2**64-1)

    def hashing(self, board):
        hashing_value = 0
        for i in range(self.S):
            piece = None
            if i <= len(board) - 1:
                if board[i] in self.val:
                    piece = self.val.index(board[i])
            if(piece is not None):
                hashing_value ^= self.zobristnum[i][piece]

        # tail = int(math.log2(self.nprocs))
        # #print (tail)
        # head = int(64-math.log2(self.nprocs))
        # #print (head)
        # hash_key = format(hashing_value, '064b')[0:head]
        # hash_key = int(hash_key, 2)
        # core_dest = format(hashing_value, '064b')[-tail:]
        # core_dest = int(core_dest, 2)

        hash_key = hashing_value
        core_dest = (hashing_value >> self.hash_index_bits) % self.nprocs

        return hash_key, core_dest

    def insert(self, item):
        hash, _ = self.hashing(item.key)
        if self.hashTable.get(hash) is None:
            self.hashTable.setdefault(hash, [])
            self.hashTable[hash].append(item)
        else:
            for i, it in enumerate(self.hashTable[hash]):
                if it.key == item.key:
                    del self.hashTable[hash][i]
            self.hashTable[hash].append(item)

    def search_table(self, key):
        hash, _ = self.hashing(key)
        if self.hashTable.get(hash) is None:
            return None
        else:
            for i, it in enumerate(self.hashTable[hash]):
                if it.key == key:
                    return it.value
        return None
