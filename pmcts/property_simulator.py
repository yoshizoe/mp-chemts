from keras.preprocessing import sequence
from Gaussian_DFT.RDKitText import tansfersdf
from Gaussian_DFT.SDF2GauInput import GauTDDFT_ForDFT
from Gaussian_DFT.GaussianRunPack import GaussianDFTRun
from pmcts import sascorer
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import networkx as nx

class simulator:
    """
    logp property
    """
    def __init__(self, property):
        self.property=property
        #print (self.property)
        if self.property=="logP":
            self.val=['\n', '&', 'C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[C@@H]',
                'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/',
                '[nH]', '[NH3+]', '4', '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '5',
                '[N-]', '[n+]', '[S@@]', '[S-]', '6', '7', 'I', '[n-]', 'P', '[OH+]', '[NH-]',
                '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]', '[o+]', '[CH2-]', '[CH-]',
                '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '8', '[S@@+]']
            self.max_len=82
        if self.property=="wavelength":
            self.val=['\n', '&', 'C', '[C@@H]', '(', 'N', ')', 'O', '=', '1', '/', 'c', 'n', '[nH]',
                '[C@H]', '2', '[NH]', '[C]', '[CH]', '[N]', '[C@@]', '[C@]', 'o', '[O]', '3', '#',
                '[O-]', '[n+]', '[N+]', '[CH2]', '[n]']
            self.max_len=42

    def run_simulator(self, new_compound, rank):
        if self.property=="logP":
            score,mol=self.logp_evaluator(new_compound, rank)
        if self.property=="wavelength":
            score,mol=self.wavelength_evaluator(new_compound, rank)
        return score, mol

    def logp_evaluator(self, new_compound, rank):
        ind=rank
        try:
            m = Chem.MolFromSmiles(str(new_compound[0]))
        except BaseException:
            m = None
        if m is not None:
            try:
                logp = Descriptors.MolLogP(m)
            except BaseException:
                logp = -1000
            SA_score = -sascorer.calculateScore(MolFromSmiles(new_compound[0]))
            cycle_list = nx.cycle_basis(
                nx.Graph(
                    rdmolops.GetAdjacencyMatrix(
                        MolFromSmiles(
                            new_compound[0]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([len(j) for j in cycle_list])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6
            cycle_score = -cycle_length
            SA_score_norm = SA_score  # (SA_score-SA_mean)/SA_std
            logp_norm = logp  # (logp-logP_mean)/logP_std
            cycle_score_norm = cycle_score  # (cycle_score-cycle_mean)/cycle_std
            score_one = SA_score_norm + logp_norm + cycle_score_norm
            score = score_one / (1 + abs(score_one))
        else:
            score = -1000 / (1 + 1000)
        return score, new_compound[0]

    def wavelength_evaluator(self, new_compound, rank):
        ind=rank
        try:
            m = Chem.MolFromSmiles(str(new_compound[0]))
        except:
            m= None
        if m!= None:
            stable = tansfersdf(str(new_compound[0]),ind)
            if stable == 1.0:
                try:
                    SDFinput = 'CheckMolopt'+str(ind)+'.sdf'
                    calc_sdf = GaussianDFTRun('B3LYP', '3-21G*', 1, 'uv homolumo', SDFinput, 0)
                    outdic = calc_sdf.run_gaussian()
                    wavelength = outdic['uv'][0]
                except:
                    wavelength = None
            else:
                wavelength = None
            if wavelength != None and wavelength != []:
                wavenum = wavelength[0]
                gap = outdic['gap'][0]
                lumo = outdic['gap'][1]
                score = 0.01*wavenum/(1+0.01*abs(wavenum))
            else:
                score = -1
        else:
            score = -1
        return score, new_compound[0]
