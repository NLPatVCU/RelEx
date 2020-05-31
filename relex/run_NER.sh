echo "MedaCy results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/final_predictions/'

echo "mimic 200 sentence"
python demo_test.py mimic 200 single
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

echo "mimic 300 sentence"
python demo_test.py mimic 300 single
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

echo "glove 200 sentence"
python demo_test.py glove 200 single
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

echo "glove 300 sentence"
python demo_test.py glove 300 single
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

echo "mimic 200 segment"
python demo_test.py mimic 200 segment
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

echo "mimic 300 segment"
python demo_test.py mimic 300 segment
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

echo "glove 200 segment"
python demo_test.py glove 200 segment
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

echo "glove 300 segment"
python demo_test.py glove 300 segment
python RelEx_NN/evaluation/refine.py
echo "RelEx results"
python RelEx_NN/evaluation/BRAT_evaluator.py '../data/test/' '../Predictions/refined_predictions/'

