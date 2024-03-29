# GraphCDD
This code is the implementation of GraphCDD

GraphCDD: Predicting circRNA-drug resistance associations based on a multimodal graph representation learning framework
#### Requirements

* python (tested on version 3.7.11)  
* pytorch (tested on version 1.6.0)  
* torch-geometric (tested on version 2.0.2)  
* numpy (tested on version 1.21.2)  
* scikit-learn(tested on version 1.0.2)  

#### Quick start

To reproduce our results:  
Run code\main.py to RUN GraphCDD.  

#### Folder

* code: Model code of GraphCDD.  
* datasets: Data required by GraphCDD.  
* results: Results of GraphCDD run.




#### Data description
* allpair.csv: all pairs of circRNAs and drug resistance  
* circ4.csv: circRNA integrated similarity
* CSS.csv: circRNA sequence similarity
* dis.csv: disease integrated similarity
* DSS.csv: disease semantic similarity
* drug.csv: drug integrated similarity
* MTS.csv: molecular structure similarity
* circ_dis.csv: circRNA-disease association matrix   
* circ_drug.csv: circRNA-drug resistance association matrix  
* drug_dis.csv: disease-drug resistance association matrix   
* circRNAname.xlsx: list of circRNA names  
* drugname.xlsx: list of drug names  
* diseasename.xlsx: list of disease names  
* NoncoRNA.xls: data used in the casestudy

#### Contacts

If you have any questions or comments, please feel free to email Ziqiang Liu(liuzq_dlmu@163.com) 



