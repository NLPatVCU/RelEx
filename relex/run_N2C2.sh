#echo "mimic 200 sentence"
#python experiments.py mimic 200 single True '../Predictions/Rel_Predictions/initial/MIMIC_200/'
#
#echo "mimic 300 sentence"
#python experiments.py mimic 300 single True '../Predictions/Rel_Predictions/initial/MIMIC_300/'
#
#echo "glove 200 sentence"
#python experiments.py glove 200 single True '../Predictions/Rel_Predictions/initial/GloVe_200/'
#
#echo "glove 300 sentence"
#python experiments.py glove 300 single True '../Predictions/Rel_Predictions/initial/GloVe_300/'


#echo "mimic 200 segment"
#python experiments.py mimic 200 segment True '../Predictions/Rel_Predictions/initial/MIMIC_200/'
#
#echo "mimic 300 segment"
#python experiments.py mimic 300 segment True '../Predictions/Rel_Predictions/initial/MIMIC_300/'
#
#echo "glove 200 segment"
#python experiments.py glove 200 segment True '../Predictions/Rel_Predictions/initial/GloVe_200/'
#
#echo "glove 300 segment"
#python experiments.py glove 300 segment True '../Predictions/Rel_Predictions/initial/GloVe_300/'


python utils/re_number.py '../Predictions/Rel_Predictions/initial/GloVe_200/' '../Predictions/Rel_Predictions/final/GloVe_200/'
python utils/re_number.py '../Predictions/Rel_Predictions/initial/GloVe_300/' '../Predictions/Rel_Predictions/final/GloVe_300/'
python utils/re_number.py '../Predictions/Rel_Predictions/initial/MIMIC_200/' '../Predictions/Rel_Predictions/final/MIMIC_200/'
python utils/re_number.py '../Predictions/Rel_Predictions/initial/MIMIC_300/' '../Predictions/Rel_Predictions/final/MIMIC_300/'
#python utils/re_number.py '../Predictions/Rel_Predictions/initial/sample/' '../Predictions/Rel_Predictions/final/sample/'


#python RelEx_NN/evaluation/BRAT_evaluator.py '../data/train/' '../Predictions/Rel_Predictions/final/MIMIC_200/'
#python RelEx_NN/evaluation/BRAT_evaluator.py '../data/train/' '../Predictions/Rel_Predictions/final/MIMIC_300/'
#python RelEx_NN/evaluation/BRAT_evaluator.py '../data/train/' '../Predictions/Rel_Predictions/final/GloVe_200/'
#python RelEx_NN/evaluation/BRAT_evaluator.py '../data/train/' '../Predictions/Rel_Predictions/final/GloVe_300/'

