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
#
#
#echo "mimic 200 segment"
#python demo_test.py mimic 200 segment True '../Predictions/Rel_Predictions/initial/MIMIC_200/'
#
#echo "mimic 300 segment"
#python demo_test.py mimic 300 segment True '../Predictions/Rel_Predictions/initial/MIMIC_300/'
#
#echo "glove 200 segment"
##python demo_test.py glove 200 segment
#python demo_test.py glove 200 segment True '../Predictions/Rel_Predictions/initial/GloVe_200/'

#echo "glove 300 segment"
#python demo_test.py glove 300 segment
#python demo_test.py glove 300 segment True '../Predictions/Rel_Predictions/initial/GloVe_300/'

#echo "chem 200 segment"
##python demo_test.py glove 200 segment
#python demo_test.py glove 200 segment True '../Predictions/Rel_Predictions/initial/Chem_200/'

#python utils/re_number.py '../Predictions/Rel_Predictions/initial/GloVe_200/' '../Predictions/Rel_Predictions/final/GloVe_200/'
#python utils/re_number.py '../Predictions/Rel_Predictions/initial/GloVe_300/' '../Predictions/Rel_Predictions/final/GloVe_300/'
#python utils/re_number.py '../Predictions/Rel_Predictions/initial/MIMIC_200/' '../Predictions/Rel_Predictions/final/MIMIC_200/'
#python utils/re_number.py '../Predictions/Rel_Predictions/initial/MIMIC_300/' '../Predictions/Rel_Predictions/final/MIMIC_300/'
#python CLEF/re_number.py '../Predictions/Rel_Predictions/initial/Chem_200/' '../Predictions/Rel_Predictions/final/Chem_200/'

#python CLEF/convert_back.py '../Predictions/Rel_Predictions/final/Chem_200/' '../Predictions/Rel_Predictions/Chem_200/'

#python CLEF/CLEF_BRAT_evaluator.py '../CLEF/key/' '../CLEF/final/GloVe_200/'
#python CLEF/CLEF_BRAT_evaluator.py '../CLEF/key/' '../CLEF/final/GloVe_300/'
python CLEF/CLEF_BRAT_evaluator.py '../CLEF/key/' '../CLEF/final/Chem_200/'

#python CLEF/append_relation.py '../CLEF/Predictions/final/GloVe_200/' '../CLEF/eval/GloVe_200/'
#python CLEF/append_relation.py '../CLEF/Predictions/final/Chem_200/' '../CLEF/eval/Chem_200/'




#python CLEF/extract_relation.py '../Predictions/Rel_Predictions/initial/GloVe_300/' '../Predictions/Rel_Predictions/initial/sample/'
#python CLEF/extract_relation.py '../data/test/' '../data/sample/'




