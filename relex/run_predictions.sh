#echo "mimic 200 sentence"
#python demo_test.py mimic 200 single True '../Predictions/MedaCy /MIMIC_200/'

#echo "mimic 300 sentence"
#python demo_test.py mimic 300 single True '../Predictions/MedaCy /MIMIC_300/'

#echo "glove 200 sentence"
#python demo_test.py glove 200 single True '../Predictions/MedaCy /Glove_200/'

#echo "glove 300 sentence"
#python demo_test.py glove 300 single True '../Predictions/MedaCy /Glove_300/'

echo "mimic 200 sentence"
python demo_test.py mimic 200 segment True '../Predictions/MedaCy /MIMIC_200/'

echo "mimic 300 sentence"
python demo_test.py mimic 300 segment True '../Predictions/MedaCy /MIMIC_300/'

echo "glove 200 sentence"
python demo_test.py glove 200 segment True '../Predictions/MedaCy /Glove_200/'

echo "glove 300 sentence"
python demo_test.py glove 300 segment True '../Predictions/MedaCy /Glove_300/'


python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/MedaCy /MIMIC_200/'
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/MedaCy /MIMIC_300/'
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/MedaCy /Glove_200/'
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/MedaCy /Glove_300/'