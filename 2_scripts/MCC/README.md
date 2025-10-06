Scripts here are used for Binary classification, VERCONF, VERCOOP, MATCONF, MATCOOP.
Examples for MCC commands:
1- python3 MCC_Strategies/AL_top_confedence.py --output MCC_Logs/AR_EN/Top_Confidence --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
2- python3 MCC_Strategies/AL_Core_set.py --output MCC_Logs/AR_EN/core_set --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
3- python3 MCC_Strategies/AL_monte_carlo.py --output MCC_Logs/AR_EN/Monte_Carlo --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
4- python3 MCC_Strategies/AL_max_entropy.py --output MCC_Logs/AR_EN/Max_Entropy --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
5- python3 MCC_Strategies/AL_Margin_Sample.py --output MCC_Logs/AR_EN/Margin_Sampling --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
6- python3 MCC_Strategies/AL_Stacking.py --output MCC_Logs/AR_EN/Stacking --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
7- python3 MCC_Strategies/AL_Stacking_Union.py --output MCC_Logs/AR_EN/Stacking_Union --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
8- python3 MCC_Strategies/fullData.py --output MCC_Logs/AR_EN/FullDate --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
9- python3 MCC_Strategies/baseline.py --output MCC_Logs/AR_EN/BaseLine --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
10- python3 MCC_Strategies/AL_Bald_Sampling.py --output MCC_Logs/AR_EN/BALD --folder Datasets/MCC_AR_EN_DEEPL --pretrain Downstream_Models/English/ConfliBERT-cont-cased
Where:
MCC_Strategies/AL_top_confedence.py: is the type of Active learning 
--output MCC_Logs/AR_EN/Top_Confidence: is the name of the output file
--folder Datasets/MCC_AR_EN_DEEPL: Where the data set is saved
--pretrain Downstream_Models/English/ConfliBERT-cont-cased: The model used for fine tuning

Note:
baseline.py is used to randomly select 20% of the data. This is used to compare against running the same budget of samples without Active learning.
fullData.py uses the full data set to fine tune the model. We use this to mesaure the gain we get from using AL
