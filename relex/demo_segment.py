from data import Dataset
from segment import Segmentation

# path to the dataset
# sample_train = Dataset('../data/sample/train')
# sample_train = Dataset('../data/medacy')
# sample_train = Dataset('../data/train')
# predictions = '../Predictions/sample_pred/'
# predictions = '../Predictions/final_predictions/'
sample_train = Dataset('../data/test')

'''
Running instructions: 
    1. Run each category of the labels separately. Comment out the rest when running 
    2. Add the entities in the category to the rel_labels list
    3. To enable the no_relation, add the label name to the no_rel_label list, if not do not pass the list as an argument

'''
# To extract the problem - test relations object
# rel_labels = ['problem', 'test']
# no_rel_label = ['NTeP']

# To extract the problem - treatment relations object
# rel_labels = ['problem', 'treatment']
# no_rel_label = ['NTrP']

# To extract the problem - problem relations object
# rel_labels = ['problem']
# no_rel_label = ['NPP']

# rel_labels = ['Chemical']
# no_rel_label = ['NoReact']

# rel_labels = ['Drug','Form']
no_rel_label = ['No-Relation']

#N2C2 data entities
# rel_labels = ['Drug', 'Form']
# rel_labels = ['Drug', 'Reason', 'ADE', 'Route', 'Frequency', 'Duration', 'Strength', 'Form', 'Dosage']
# rel_labels = ['Drug', 'Symptom', 'Route', 'Frequency', 'Duration', 'Strength', 'Form', 'Dosage']

#END data entities
# rel_labels = ['Action', 'Active_ingredient', 'Adverse_reaction', 'amount_misc', 'amount_unit', 'Assay', 'AUC', 'Capping_agent', 'Cell_type', 'Characterization_method', 'Clearance', 'Cmax', 'Coadministered_drug', 'Company',
#               'Complex', 'Concentration', 'condition_unit', 'Core_composition', 'DNA', 'Dose', 'Duration', 'Elimination_half_life', 'Encapsulation_efficiency', 'Endpoint', 'FDA_approval_date',
#               'Frequency', 'gas', 'Gene', 'Group_name', 'Group_size', 'IC50', 'Inactive_ingredient', 'Indication', 'Instrument', 'Instrument_setting', 'Intermediate', 'LC50', 'Lipid', 'Loading',
#               'LOAEL', 'LogP', 'Mass', 'material', 'material_descriptor', 'meta', 'Metabolite', 'Method', 'Model', 'Molecular_weight', 'Moles', 'Nanoparticle', 'NOAEL', 'nonrecipe_material', 'number',
#               'operation', 'Organism', 'Other_chemical', 'Particle_diameter', 'Particle_dimension', 'Particle_shape', 'pH', 'phase', 'Plasma_half_life', 'Precursor', 'Preexisting_disease', 'Pressure',
#               'property_misc', 'Protein', 'Purity', 'Reducing_agent', 'RNA', 'Route_of_administration', 'Sample_size', 'Seed_solution', 'Sex', 'Shell_composition', 'Species', 'Solvent', 'Strain',
#               'Surface_area', 'Surface_coating', 'Surfactant', 'Synthesis_method', 'target', 'Targeting_molecule', 'Temperature', 'Test_article', 'Time', 'tmax', 'Trade_name', 'unspecified_material',
#               'US_patents', 'Vehicle', 'Volume', 'Volume_of_distribution', 'Water_solubility', 'Yield', 'Zeta_potential']

#CLEF Data entities
rel_labels= ['REACTION_STEP','TIME', 'TEMPERATURE', 'YIELD_OTHER', 'YIELD_PERCENT']
# rel_labels = ['WORKUP', 'REACTION_STEP', 'EXAMPLE_LABEL', 'REACTION_PRODUCT', 'STARTING_MATERIAL', 'REAGENT_CATALYST', 'SOLVENT', 'OTHER_COMPOUND', 'TIME', 'TEMPERATURE', 'YIELD_OTHER', 'YIELD_PERCENT']
# rel_labels = ['Action', 'Duration', 'Frequency', 'Volume', 'Solvent','Concentration','Surfactant']

# to extract the segments from the dataset
seg_sampleTrain = Segmentation(sample_train, rel_labels, test= True, generalize=False)
# seg_sampleTrain = Segmentation(sample_train, rel_labels,  no_rel_label, generalize=False, write_Predictions=True, prediction_folder=predictions)
# seg_sampleTrain = Segmentation(sample_train, rel_labels,  no_rel_label, generalize=False)
# seg_sampleTrain = Segmentation(sample_train, rel_labels, generalize=False)
# seg_sampleTrain = Segmentation(sample_train, rel_labels, no_rel_label, same_entity_relation = True)

#print for testing purposes
sample_track = seg_sampleTrain.segments['track']
sample_label = seg_sampleTrain.segments['label']
# print(sample_track)
# print(sample_label)
