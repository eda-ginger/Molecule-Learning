# RDKit libraries
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings

# Open Babel libraries
from openbabel import pybel, openbabel
import numpy as np

# Define graph featurizer class
class Graph_Featurizer(object):

    """
    A class for generating graph-based features for molecular structures,
    including nodes and edges for protein-ligand complexes. The class supports
    various atomic and molecular properties such as van der Waals radii,
    electronegativity, polarizability, and more.

    Attributes:
        surface_features_bool (bool): Indicates whether surface features, including hydrogens, should be computed.
        atom_classes (list): List of atomic numbers for specific elements (e.g., C, O, N, etc.).
        ATOM_CODES (dict): Mapping of atomic numbers to one-hot encoding indices.
        VDW_RADII (dict): Van der Waals radii for atoms (in pm).
        ELECTRONEGATIVITY (dict): Pauling electronegativity values for atoms.
        POLARIZABILITY (dict): Dipole polarizability values for atoms.
        AMINO_ACID_TYPES (dict): Classification of amino acids by type (e.g., hydrophobic, polar, etc.).
        AMINO_ACID_TYPE_CODES (dict): One-hot encoding for amino acid types.
    """

    # Define the class constructor
    def __init__(self, surface_features_bool=False):

        self.surface_features_bool = surface_features_bool

        # Define the atom classes for C, O, N, S, F, P, Cl, Br, B, I, and other
        self.atom_classes = [(6), (8), (7), (16), (9), (15), (17), (35), (5), (53), (-1)]

        # add hydrogen if computing surface features
        if surface_features_bool:
            self.atom_classes.insert(0, (1))

        self.ATOM_CODES = {atomic_num: idx for idx, atomic_num in enumerate(self.atom_classes)}

        # Define the atomic Van der Waals radii (pm) 
        self.VDW_RADII = {
            5: 192,  # Boron
            6: 170,  # Carbon
            7: 155,  # Nitrogen
            8: 152,  # Oxygen
            9: 135,  # Fluorine
            15: 180, # Phosphorus
            16: 180, # Sulfur
            17: 175, # Chlorine
            35: 183, # Bromine
            53: 198, # Iodine

            26: 194,  # Fe (Iron)
            44: 207,  # Ru (Ruthenium)
            34: 190,  # Se (Selenium)
            14: 210,  # Si (Silicon)
            77: 202,  # Ir (Iridium)
            33: 185,  # As (Arsenic)
            27: 192,  # Co (Cobalt)
            23: 179,  # V (Vanadium)
            78: 209,  # Pt (Platinum)
            45: 195,  # Rh (Rhodium)
            4: 153,   # Be (Beryllium)
            76: 216,  # Os (Osmium)
            75: 217,  # Re (Rhenium)
            29: 140,  # Cu (Copper)
            51: 206,  # Sb (Antimony)
            12: 173,  # Mg (Magnesium)
            30: 139,  # Zn (Zinc)
            52: 206,  # Te (Tellurium)
            }


        # Define the electronegativity values as per the Pauling scale
        self.ELECTRONEGATIVITY = {
            5: 2.04,   # Boron
            6: 2.55,   # Carbon
            7: 3.04,   # Nitrogen
            8: 3.44,   # Oxygen
            9: 3.98,   # Fluorine
            15: 2.19,  # Phosphorus
            16: 2.58,  # Sulfur
            17: 3.16,  # Chlorine
            35: 2.96,  # Bromine
            53: 2.66,  # Iodine

            26: 1.83,  # Fe (Iron)
            44: 2.20,  # Ru (Ruthenium)
            34: 2.55,  # Se (Selenium)
            14: 1.90,  # Si (Silicon)
            77: 2.20,  # Ir (Iridium)
            33: 2.18,  # As (Arsenic)
            27: 1.88,  # Co (Cobalt)
            23: 1.63,  # V (Vanadium)
            78: 2.28,  # Pt (Platinum)
            45: 2.28,  # Rh (Rhodium)
            4: 1.57,  # Be (Beryllium)
            76: 2.20,  # Os (Osmium)
            75: 1.90,  # Re (Rhenium)
            29: 1.90,  # Cu (Copper)
            51: 2.05,  # Sb (Antimony)
            12: 1.31,  # Mg (Magnesium)
            30: 1.65,  # Zn (Zinc)
            52: 2.10,  # Te (Tellurium)
            }


        # Define the static dipole polarizability values of neutral elements as defined 
        self.POLARIZABILITY = {
            5: 20.50,   # Boron
            6: 11.30,   # Carbon
            7: 7.40,   # Nitrogen
            8: 5.30,   # Oxygen
            9: 3.74,   # Fluorine
            15: 25.10,  # Phosphorus
            16: 19.40,  # Sulfur
            17: 14.60,  # Chlorine
            35: 21.00,  # Bromine
            53: 32.90,  # Iodine

            26: 62.00,  # Fe (Iron)
            44: 72.00,  # Ru (Ruthenium)
            34: 28.90,  # Se (Selenium)
            14: 37.30,  # Si (Silicon)
            77: 54.00,  # Ir (Iridium)
            33: 30.00,  # As (Arsenic)
            27: 55.00,  # Co (Cobalt)
            23: 87.00,  # V (Vanadium)
            78: 48.00,  # Pt (Platinum)
            45: 66.00,  # Rh (Rhodium)
            4: 37.74,  # Be (Beryllium)
            76: 57.00,  # Os (Osmium)
            75: 62.00,  # Re (Rhenium)
            29: 46.50,  # Cu (Copper)
            51: 43.00,  # Sb (Antimony)
            12: 71.20, # Mg (Magnesium)
            30: 38.67,  # Zn (Zinc)
            52: 38.00,  # Te (Tellurium)
                }

        # Classify each amino acid type as hydrophobic, polar, basic, or acidic
        self.AMINO_ACID_TYPES = {
            "ALA": "hydrophobic",
            "VAL": "hydrophobic",
            "LEU": "hydrophobic",
            "ILE": "hydrophobic",
            "MET": "hydrophobic",
            "PHE": "hydrophobic",
            "TYR": "hydrophobic",
            "TRP": "hydrophobic",
            "SER": "polar",
            "THR": "polar",
            "ASN": "polar",
            "GLN": "polar",
            "LYS": "basic",
            "ARG": "basic",
            "HIS": "basic",
            "ASP": "acidic",
            "GLU": "acidic",
            "CYS": "polar", # special case
            "GLY": "hydrophobic", # special case
            "PRO": "hydrophobic", # special case
            "LIG": "ligand", # case for ligand atoms
        }

        if surface_features_bool:
            self.VDW_RADII.update({1: 120}) # Hydrogen
            self.ELECTRONEGATIVITY.update({1: 2.20}) # Hydrogen
            self.POLARIZABILITY.update({1: 4.51}) # Hydrogen



        # Define the amino acid type one-hot encodings
        self.AMINO_ACID_TYPE_CODES = {
            "hydrophobic": 0,
            "polar": 1,
            "basic": 2,
            "acidic": 3,
            "ligand": 4,
        }


    # Method to encode the atomic number into a one-hot vector
    def encode_atomic_number(self, atomic_num):
        encoding = np.zeros(len(self.atom_classes))
        if atomic_num in self.ATOM_CODES:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        else:
            encoding[self.ATOM_CODES[-1]] = 1.0 # Other class
        return encoding


    # Method to encode amino acid type into a one-hot vector
    def encode_amino_acid_type(self, residue_name, graph_type='protein'):
        if graph_type == 'protein':
            type_str = self.AMINO_ACID_TYPES.get(residue_name, "unknown")
            encoding = np.zeros(len(self.AMINO_ACID_TYPE_CODES) - 1)
            if type_str in self.AMINO_ACID_TYPE_CODES:
                encoding[self.AMINO_ACID_TYPE_CODES[type_str]] = 1.0
            return encoding

        elif graph_type == 'complex':
            if residue_name == "LIG":
                type_str = "ligand"
            else:
                type_str = self.AMINO_ACID_TYPES.get(residue_name, "unknown") # Default to unknown if not found
            encoding = np.zeros(len(self.AMINO_ACID_TYPE_CODES))
            if type_str in self.AMINO_ACID_TYPE_CODES:
                encoding[self.AMINO_ACID_TYPE_CODES[type_str]] = 1.0
            return encoding


    # Method to determine if an atom is hydrophobic
    def is_hydrophobic(self, atom):
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()

        # Check if the atom is neutral
        if formal_charge != 0:
            return False

        # Neutral carbon atoms not bonded to N, O, or F
        if atomic_num == 6:  # Carbon
            bonded_elements = {neighbor.GetAtomicNum() for neighbor in openbabel.OBAtomAtomIter(atom)}
            if not (7 in bonded_elements or 8 in bonded_elements or 9 in bonded_elements):
                return True

        # Neutral sulfur in specific oxidation states (simplified check)
        elif atomic_num == 16:  # Sulfur
            hybridization = atom.GetHyb()
            if hybridization in [2, 3]:  # Simplified logic for SH or sulfur with sp2/sp3 hybridization
                return True

        # Neutral halogens (Cl, Br, I)
        elif atomic_num in [17, 35, 53]:  # Chlorine, Bromine, Iodine
            return True

        return False


    # Method to calculate the features for a molecule
    def get_node_features(self, molecule, source='ligand', complex_bool=False):
        # print('We are in the get_node_features method')
        node_features = []
        coordinates = []

        for i, atom in enumerate(molecule):
            # print('We are processing atom number', i)
            if atom.atomicnum > 1 or self.surface_features_bool: # Skip hydrogen atoms unless computing surface features

                atomic_number_encoding = self.encode_atomic_number(atom.atomicnum)
                vdw_radius = self.VDW_RADII.get(atom.atomicnum, 183.32) # default to mean value
                formal_charge = atom.OBAtom.GetFormalCharge()
                partial_charge = atom.OBAtom.GetPartialCharge()
                electronegativity = self.ELECTRONEGATIVITY.get(atom.atomicnum, 2.29) # default to mean value
                polarizability = self.POLARIZABILITY.get(atom.atomicnum, 39.12) # default to mean value
                hydrophobic = 1.0 if self.is_hydrophobic(atom.OBAtom) else 0.0
                aromatic = 1.0 if atom.OBAtom.IsAromatic() else 0.0
                acceptor = 1.0 if atom.OBAtom.IsHbondAcceptor() else 0.0
                donor = 1.0 if atom.OBAtom.IsHbondDonor() else 0.0
                ring = 1.0 if atom.OBAtom.IsInRing() else 0.0
                hybridization = atom.OBAtom.GetHyb()
                chirality = 1.0 if atom.OBAtom.IsChiral() else 0.0
                total_degree = len([neighbor for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom)])
                heavy_degree = sum(1 for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom) if neighbor.GetAtomicNum() > 1)
                hetero_degree = sum(1 for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom) if neighbor.GetAtomicNum() not in [1, 6])
                hydrogen_degree = sum(1 for neighbor in openbabel.OBAtomAtomIter(atom.OBAtom) if neighbor.GetAtomicNum() == 1)

                atom_coords = np.array([atom.OBAtom.GetX(), atom.OBAtom.GetY(), atom.OBAtom.GetZ()])
                coordinates.append(atom_coords)

                if complex_bool == True:
                    source_id = -1 if source == "protein" else 1
                    amino_acid_type_encoding = self.encode_amino_acid_type(atom.residue.name, 'complex') if source == "protein" else self.encode_amino_acid_type("LIG", 'complex')
                    node_features.append(
                        np.concatenate(
                        (
                            atomic_number_encoding, # C, O, N, S, F, P, Cl, Br, B, I, other
                            amino_acid_type_encoding, # hydropobic, polar, basic, acidic, ligand

                            [formal_charge],
                            [hydrophobic],
                            [aromatic],
                            [acceptor],
                            [donor],
                            [ring],
                            [hybridization],
                            [chirality],
                            [total_degree],
                            [heavy_degree],
                            [hetero_degree],
                            [hydrogen_degree],
                            [source_id],

                            [vdw_radius],
                            [partial_charge],
                            [electronegativity],
                            [polarizability],

                    )
                    )
                    )

                elif complex_bool == False:
                    if source == "protein":
                        amino_acid_type_encoding = self.encode_amino_acid_type(atom.residue.name, 'protein')
                        node_features.append(
                            np.concatenate(
                            (
                                atomic_number_encoding, # C, O, N, S, F, P, Cl, Br, B, I, other
                                amino_acid_type_encoding, # hydropobic, polar, basic, acidic

                                [formal_charge],
                                [hydrophobic],
                                [aromatic],
                                [acceptor],
                                [donor],
                                [ring],
                                [hybridization],
                                [chirality],
                                [total_degree],
                                [heavy_degree],
                                [hetero_degree],
                                [hydrogen_degree],

                                [vdw_radius],
                                [partial_charge],
                                [electronegativity],
                                [polarizability],
                        )
                        )
                        )

                    elif source == "ligand":
                        node_features.append(
                            np.concatenate(
                            (
                                atomic_number_encoding, # C, O, N, S, F, P, Cl, Br, B, I, other

                                [formal_charge],
                                [hydrophobic],
                                [aromatic],
                                [acceptor],
                                [donor],
                                [ring],
                                [hybridization],
                                [chirality],
                                [total_degree],
                                [heavy_degree],
                                [hetero_degree],
                                [hydrogen_degree],

                                [vdw_radius],
                                [partial_charge],
                                [electronegativity],
                                [polarizability],
                        )
                        )
                        )

        # Format and return the features
        node_features = np.array(node_features, dtype=np.float32)
        # print('We converted the node features to a numpy array')
        coordinates = np.array(coordinates, dtype=np.float32)
        # print('We converted the coordinates to a numpy array')
        return node_features, coordinates


    def get_bond_based_edges(self, molecule):

        # Remove hydrogens from the molecule
        molecule.OBMol.DeleteHydrogens()

        edge_idx, edge_attr = [], []
        for bond in openbabel.OBMolBondIter(molecule.OBMol):

            atom1 = bond.GetBeginAtomIdx() - 1  # OpenBabel indices start at 1, so subtract 1 for 0-indexing
            atom2 = bond.GetEndAtomIdx() - 1

            if bond.GetBeginAtom().GetAtomicNum() == 1 or bond.GetEndAtom().GetAtomicNum() == 1:
                continue  # Skip hydrogen bonds

            bond_order = bond.GetBondOrder()

            atom1_coords = np.array([bond.GetBeginAtom().GetX(), bond.GetBeginAtom().GetY(), bond.GetBeginAtom().GetZ()])
            atom2_coords = np.array([bond.GetEndAtom().GetX(), bond.GetEndAtom().GetY(), bond.GetEndAtom().GetZ()])

            distance = np.linalg.norm(atom1_coords - atom2_coords)

            aromatic = 1.0 if bond.IsAromatic() else 0.0
            ring = 1.0 if bond.IsInRing() else 0.0

            atom1_electronegativity = self.ELECTRONEGATIVITY.get(bond.GetBeginAtom().GetAtomicNum(), 0)
            atom2_electronegativity = self.ELECTRONEGATIVITY.get(bond.GetEndAtom().GetAtomicNum(), 0)
            electronegativity_difference = np.abs(atom1_electronegativity - atom2_electronegativity)

            atom1_charge = bond.GetBeginAtom().GetPartialCharge()
            atom2_charge = bond.GetEndAtom().GetPartialCharge()

            electrostatic_energy = (atom1_charge * atom2_charge) / distance**2 # TODO: Add this as a feature

            edge_idx.append((atom1, atom2))
            edge_idx.append((atom2, atom1))
            edge_attr.append([bond_order, distance, aromatic, ring, electronegativity_difference, electrostatic_energy])
            edge_attr.append([bond_order, distance, aromatic, ring, electronegativity_difference, electrostatic_energy])

        return edge_idx, edge_attr


    def get_distance_based_edges(self, protein, ligand, distance_threshold=4.5): # 4.5 A corresponds to hydrophobic threshold deined by ProLIF

        # remove hydrogens from the protein and ligand
        protein.OBMol.DeleteHydrogens()
        ligand.OBMol.DeleteHydrogens()

        # Get the coordinates and electronegativity of the protein atoms
        protein_coords, protein_electronegativities, protein_charges = [], [], []
        for atom in openbabel.OBMolAtomIter(protein.OBMol):
            if atom.GetAtomicNum() == 1:
                continue  # Skip hydrogen atoms

            protein_coords.append(np.array([atom.GetX(), atom.GetY(), atom.GetZ()]))
            protein_electronegativities.append(self.ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 0))
            protein_charges.append(atom.GetPartialCharge())

        # Get the coordinates and electronegativity of the ligand atoms
        ligand_coords, ligand_electronegativities, ligand_charges = [], [], []
        for atom in openbabel.OBMolAtomIter(ligand.OBMol):
            if atom.GetAtomicNum() == 1:
                continue # Skip hydrogen atoms

            ligand_coords.append(np.array([atom.GetX(), atom.GetY(), atom.GetZ()]))
            ligand_electronegativities.append(self.ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 0))
            ligand_charges.append(atom.GetPartialCharge())

        protein_coords = np.array(protein_coords)
        ligand_coords = np.array(ligand_coords)

        # Calculate the pairwise distances between protein and ligand atoms
        distances = np.linalg.norm(protein_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :], axis=2)
        protein_indices, ligand_indices = np.where(distances <= distance_threshold)

        # Offset ligand atom indices by the number of protein atoms
        edge_idx, edge_attr = [], []

        for i, j in zip(protein_indices, ligand_indices):
            ligand_index = j + len(protein_coords)

            distance = distances[i, j]
            electronegativity_difference = np.abs(protein_electronegativities[i] - ligand_electronegativities[j])


            electrostatic_energy = (protein_charges[i] * ligand_charges[j]) / distance**2 # TODO: Add this as a feature

            edge_idx.append((i, ligand_index))
            edge_idx.append((ligand_index, i))

            edge_attr.append([0, distance, 0, 0, electronegativity_difference, electrostatic_energy])
            edge_attr.append([0, distance, 0, 0, electronegativity_difference, electrostatic_energy])

        return edge_idx, edge_attr

    def get_protein_ligand_complex_edges(self, protein, ligand):

        # Get bond-based edges for protein and ligand
        protein_edges, protein_edge_attrs = self.get_bond_based_edges(protein)
        ligand_edges, ligand_edge_attrs = self.get_bond_based_edges(ligand)

        # Offset ligand atom indices
        ligand_edges_offset = [(i + protein.OBMol.NumAtoms(), j + protein.OBMol.NumAtoms()) for i, j in ligand_edges]

        # Assign binary interaction labels
        protein_edge_attrs = [attr + [0] for attr in protein_edge_attrs]  # 0 for protein-protein
        ligand_edge_attrs = [attr + [0] for attr in ligand_edge_attrs]  # 0 for ligand-ligand

        all_edges = protein_edges + ligand_edges_offset
        all_edge_attrs = protein_edge_attrs + ligand_edge_attrs

        # Get distance-based edges between protein and ligand
        protein_ligand_edges, protein_ligand_attrs = self.get_distance_based_edges(protein, ligand)
        protein_ligand_attrs = [attr + [1] for attr in protein_ligand_attrs]  # 1 for protein-ligand


        # Combine all edges
        all_edges += protein_ligand_edges
        all_edge_attrs += protein_ligand_attrs

        return all_edges, all_edge_attrs
     

def get_graphs(connected_featurizer=None, unconnected_featurizer=None, protein_pocket_path=None, ligand_mol2_path=None, protein_path=None):

    """
    Generates graph-based features and attributes for a protein pocket, ligand,
    protein-ligand complex, and unconnected protein.

    Args:
        connected_featurizer (Graph_Featurizer): An instance of the `Graph_Featurizer`
            class for generating connected graph features (e.g., bonds and interactions).
        unconnected_featurizer (Graph_Featurizer): An instance of the `Graph_Featurizer`
            class for generating unconnected features (e.g., atomic-level features only).
        protein_pocket_path (str): Path to the protein pocket structure file (PDB format).
        ligand_mol2_path (str): Path to the ligand structure file (MOL2 format).
        protein_path (str): Path to the full protein structure file (PDB format).

    Returns:
        - protein_coords (np.ndarray): Coordinates of protein pocket atoms.
        - protein_features (np.ndarray): Features of protein pocket atoms.
        - prot_edges (list of tuples): Bond-based edges for the protein pocket.
        - prot_attrs (list of lists): Attributes of bond-based edges for the protein pocket.
        - ligand_coords (np.ndarray): Coordinates of ligand atoms.
        - ligand_features (np.ndarray): Features of ligand atoms.
        - lig_edges (list of tuples): Bond-based edges for the ligand.
        - lig_attrs (list of lists): Attributes of bond-based edges for the ligand.
        - complex_coords (np.ndarray): Coordinates of protein-ligand complex atoms.
        - complex_features (np.ndarray): Features of protein-ligand complex atoms.
        - complex_edges (list of tuples): Edges (both bond-based and distance-based)
            for the protein-ligand complex.
        - complex_attrs (list of lists): Attributes of edges for the protein-ligand complex.
        - prot_withH_coords (np.ndarray): Coordinates of protein atoms (with hydrogens).
        - prot_withH_atom_types (np.ndarray): One-hot encoded atom types for the protein
            (with hydrogens).
        - prot_withH_features (np.ndarray): Features of protein atoms (excluding atom types)
            for the protein (with hydrogens).
        """

    # Process protein pocket
    prot = next(pybel.readfile("pdb",protein_pocket_path))
    protein_features, protein_coords = connected_featurizer.get_node_features(prot, source='protein', complex_bool=False)
    prot_edges, prot_attrs = connected_featurizer.get_bond_based_edges(prot)

    # Process ligand
    lig = next(pybel.readfile("mol2", ligand_mol2_path))
    ligand_features, ligand_coords = connected_featurizer.get_node_features(lig, source='ligand', complex_bool=False)
    lig_edges, lig_attrs = connected_featurizer.get_bond_based_edges(lig)

    # Process protein-ligand complex
    protein_features_complex, protein_coords_complex = connected_featurizer.get_node_features(prot, source='protein', complex_bool=True)
    ligand_features_complex, ligand_coords_complex = connected_featurizer.get_node_features(lig, source='ligand', complex_bool=True)
    complex_features = np.concatenate((protein_features_complex, ligand_features_complex), axis=0)
    complex_coords = np.concatenate((protein_coords_complex, ligand_coords_complex), axis=0)
    complex_edges, complex_attrs = connected_featurizer.get_protein_ligand_complex_edges(prot, lig)

    # Process unconnected protein
    prot_withH = next(pybel.readfile("pdb", protein_path))
    prot_withH_features, prot_withH_coords = unconnected_featurizer.get_node_features(prot_withH, source='protein', complex_bool=False)
    prot_withH_coords, prot_withH_atom_types, prot_withH_features = np.array(prot_withH_coords), np.array(prot_withH_features[:, :12]), np.array(prot_withH_features[:, 12:])

    return protein_coords, protein_features, prot_edges, prot_attrs, ligand_coords, ligand_features, lig_edges, lig_attrs, complex_coords, complex_features, complex_edges, complex_attrs, prot_withH_coords, prot_withH_atom_types, prot_withH_features

