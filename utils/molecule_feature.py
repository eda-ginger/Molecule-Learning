import os
import os.path as osp
import pandas as pd
import re
from typing import Callable, Dict, Optional, Tuple, Union, List, Any

import torch

from torch_geometric.data import InMemoryDataset, download_url, extract_gz
# from torch_geometric.utils import from_smiles

from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
from torch_geometric.data import Data

import warnings; warnings.filterwarnings('ignore') ## 경고 무시

# Atom feature sizes (From KANO/chempromp/feature/featurization.py)
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)), 
    'degree': [0, 1, 2, 3, 4, 5], 
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# from chem/loader.py
EDGE_FEATURES = {
    'possible_bonds' : [
        Chem.rdchem.BondType.UNSPECIFIED,
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.QUADRUPLE,
        Chem.rdchem.BondType.QUINTUPLE,
        Chem.rdchem.BondType.HEXTUPLE,
        Chem.rdchem.BondType.ONEANDAHALF,
        Chem.rdchem.BondType.TWOANDAHALF,
        Chem.rdchem.BondType.THREEANDAHALF,
        Chem.rdchem.BondType.FOURANDAHALF,
        Chem.rdchem.BondType.FIVEANDAHALF,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.IONIC,
        Chem.rdchem.BondType.HYDROGEN,
        Chem.rdchem.BondType.THREECENTER,
        Chem.rdchem.BondType.DATIVEONE,
        Chem.rdchem.BondType.DATIVE,
        Chem.rdchem.BondType.DATIVEL,
        Chem.rdchem.BondType.DATIVER,
        Chem.rdchem.BondType.OTHER,
        Chem.rdchem.BondType.ZERO,
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def feature_to_onehot(value: int, choices: List[int]) -> List[int]:
    r"""
    From KANO/chempromp/feature/featurization.py > onek_encoding_unk()
    Creates a one-hot encoding.

    value: The value for which the encoding should be one.
    choices: A list of possible values
    return: A one-hot encoding of the value in a laist of length len(choices) + 1
    If value is not in the list of choices, then the final emement in the encoding is 1
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    r"""
    From KANO/chempromp/feature/featurization.py > atom_features()
    Builds a feature vector for an atom

    node feature
    : 원자 번호, degree, formalCharge, 카이랄성, 수소 수, Hybridization, 방향족 여부, 질량
    방향족 여부와 질량 제외 one-hot으로 입력
    """
    features = feature_to_onehot(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
            feature_to_onehot(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            feature_to_onehot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            feature_to_onehot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
            feature_to_onehot(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
            feature_to_onehot(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01] # scaled to about the same range as other features
    return features           
    

def atom_features_simple(atom: Chem.rdchem.Atom) -> List[Union[bool, int, float]]:
    r"""
    node feature
    : 원자 번호, 카이랄성
    """
    # features = feature_to_onehot(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
    #         feature_to_onehot(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag'])
    allowable_features = {
        'possible_atomic_num_list' : list(range(1, 119)),
        'possible_chirality_list' : [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER,        
            Chem.rdchem.ChiralType.CHI_ALLENE,
            Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
            Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
            Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
        ]}
                
    atom_feature = [allowable_features['possible_atomic_num_list'].index(
        atom.GetAtomicNum())] + [allowable_features[
        'possible_chirality_list'].index(atom.GetChiralTag())]
    return atom_feature


def smiles_to_feature(smiles: str, dim3d: str = False, 
                      with_hydrogen: bool = False,
                      kekulize: bool = False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)
    
    if dim3d:
        mol = Chem.AddHs(mol)
        mol = Chem.EmbedMolecule(mol)
        if mol == -1:
            rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        pos = np.array([conf.GetAtomPosition(idx) for idx, symbol in atom_info])
        graph_data.pos = pos

    xs: List[List[int]] = []
    tmp = 0
    for atom in mol.GetAtoms():
        current_atom_feat = atom_features(atom)
        xs.append(current_atom_feat)
        
    x = torch.tensor(xs, dtype=torch.long).view(-1, 133)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_feature = [EDGE_FEATURES['possible_bonds'].index(bond.GetBondType())] + [EDGE_FEATURES['possible_bond_dirs'].index(bond.GetBondDir())]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [edge_feature, edge_feature]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


def mol_to_feature(mol: Chem.Mol) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    xs: List[List[int]] = []
    tmp = 0
    for atom in mol.GetAtoms():
        current_atom_feat = atom_features(atom)
        xs.append(current_atom_feat)
        
    x = torch.tensor(xs, dtype=torch.long).view(-1, 133)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_feature = [EDGE_FEATURES['possible_bonds'].index(bond.GetBondType())] + [EDGE_FEATURES['possible_bond_dirs'].index(bond.GetBondDir())]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [edge_feature, edge_feature]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def mol_to_feature_simple(mol: Chem.Mol) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    xs: List[List[int]] = []
    tmp = 0
    for atom in mol.GetAtoms():
        current_atom_feat = atom_features_simple(atom)
        xs.append(current_atom_feat)
        
    x = torch.tensor(xs, dtype=torch.long).view(-1, 2)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_feature = [EDGE_FEATURES['possible_bonds'].index(bond.GetBondType())] + [EDGE_FEATURES['possible_bond_dirs'].index(bond.GetBondDir())]

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [edge_feature, edge_feature]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 2)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class CustomMoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.org/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the :ogb:`null`
    `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"ESOL"`, :obj:`"FreeSolv"`,
            :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`, :obj:`"HIV"`,
            :obj:`"BACE"`, :obj:`"BBBP"`, :obj:`"Tox21"`, :obj:`"ToxCast"`,
            :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'

    # Format: name: (display_name, url_name, csv_name, smiles_idx, y_idx)
    names: Dict[str, Tuple[str, str, str, int, Union[int, slice]]] = {
        'esol': ('ESOL', 'delaney-processed.csv', 'delaney-processed', -1, -2),
        'freesolv': ('FreeSolv', 'SAMPL.csv', 'SAMPL', 1, 2),
        'lipo': ('Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity', 2, 1),
        'pcba': ('PCBA', 'pcba.csv.gz', 'pcba', -1, slice(0, 128)),
        'muv': ('MUV', 'muv.csv.gz', 'muv', -1, slice(0, 17)),
        'hiv': ('HIV', 'HIV.csv', 'HIV', 0, -1),
        'bace': ('BACE', 'bace.csv', 'bace', 0, 2),
        'bbbp': ('BBBP', 'BBBP.csv', 'BBBP', -1, -2),
        'tox21': ('Tox21', 'tox21.csv.gz', 'tox21', -1, slice(0, 12)),
        'toxcast':
        ('ToxCast', 'toxcast_data.csv.gz', 'toxcast_data', 0, slice(1, 618)),
        'sider': ('SIDER', 'sider.csv.gz', 'sider', 0, slice(1, 28)),
        'clintox': ('ClinTox', 'clintox.csv.gz', 'clintox', 0, slice(1, 3)),
    }

    def __init__(
        self,
        root: str,
        name: str,
        feature: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False
    ) -> None:
        self.name = name.lower()
        self.feature = feature
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.feature}.pt'

    def download(self) -> None:
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self) -> None:
        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_smiles_list = [] ## smiles 저장 추가
        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            values = line.split(',')

            smiles = values[self.names[self.name][3]]
            labels = values[self.names[self.name][4]]
            labels = labels if isinstance(labels, list) else [labels]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in labels]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("{} is removed".format(smiles))
                continue
            else:
                if mol.GetNumAtoms() < 3:
                    print("{} is removed".format(smiles))
                    continue

            if self.feature == '2D-GNN':
                data = smiles_to_feature(smiles)
            elif self.feature == '3D-GNN':
                data = smiles_to_feature(smiles)
            elif self.feature == 'FP-Morgan':
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                data = Data(x=torch.tensor(fp).view(1, -1), smiles=smiles)
            elif self.feature == 'FP-MACCS':
                fp = MACCSkeys.GenMACCSKeys(mol)
                data = Data(x=torch.tensor(fp).view(1, -1), smiles=smiles)
            elif self.feature == 'CNN':
                data = integer_label_encoding(smiles, 'drug')
            
            # ## 유효하지 않은 feature 제거
            # if data.x.shape[0] < 3: 
            #     print("{} is removed".format(data.smiles))
            #     continue 
            
            # data.y = y
            data.y = torch.nan_to_num(y, 0) ## Nan 값은 0으로 바꿔서 y값 지정
            
            # Convert 0 to -1 for binary classification (0 -> -1, 1 -> 1)
            if self.name not in ['freesolv', 'esol', 'lipo']:
                data.y = torch.where(data.y == 0, torch.tensor(-1.0), data.y)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            data_smiles_list.append(smiles) ## smiles 저장 추가

        ## smiles 저장 추가
        data_smiles_series = pd.Series(data_smiles_list) ##
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False) ##
        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'
    

CHARSMISET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64, '~': 65} # add ~: 65 

CHARISOSMILEN = 65

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


########################################################################################################################
########## Function
########################################################################################################################

import numpy as np
def integer_label_encoding(sequence, tp, max_length=100):
    """
    Integer encoding for string sequence.
    Args:
        sequence (str): Drug or Protein string sequence.
        max_length: Maximum encoding length of input string.
    """
    if tp == 'drug':
        charset = CHARSMISET
    elif tp == 'protein':
        charset = CHARPROTSET

    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            if tp == 'protein':
                letter = letter.upper()
            letter = str(letter)
            encoding[idx] = charset[letter]
        except KeyError:
            print(
                f"character {letter} does not exists in sequence category encoding, skip and treat as padding."
            )
    return Data(x=torch.from_numpy(encoding).to(torch.long).unsqueeze(dim=0))
