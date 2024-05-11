# Appunti sullo sviluppo del progetto

Il titolo di riferimento per la prima parte di studio è Microsoft. Perché:
* molto capitalizzato
* longevo
* è un titolo tech ma non subisce dinamiche troppo "strane" rispetto al normale andamento di mercato (es. Tesla)

Per prima cosa si testa la performance di un modello non trainato, semplice, che prova a predire il segno del ritorno del giorno successivo sulla base del ritorno del giorno precedente (+ segue + e - segue -).

Si riporta anche un piccolo grafico a barre per avere un'idea della distribuzione dei ritorni.

winrate detected: 0.47638123852445335

Primi test con logistic regression, aggiungendo come features giorni passati: lieve increase, troppi giorni porta a overfitting

provare a effettuare lo stesso test ma aggiungendo qualche metrica (es. moving average) -> C'è un lieve miglioramento

Nell'MLP, nei nomi dei file di dati, i numeri sono la dimensione degli hidden layer, in ordine di profondità
Primo test semplice semplice, architettura di seguito:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 20, 50)            10400     
                                                                 
 dropout (Dropout)           (None, 20, 50)            0         
                                                                 
 lstm_1 (LSTM)               (None, 20, 50)            20200     
                                                                 
 dropout_1 (Dropout)         (None, 20, 50)            0         
                                                                 
 lstm_2 (LSTM)               (None, 50)                20200     
                                                                 
 dropout_2 (Dropout)         (None, 50)                0         
                                                                 
 dense (Dense)               (None, 1)                 51        
                                                                 
=================================================================
Total params: 50,851
Trainable params: 50,851
Non-trainable params: 0
semplici 25 epoche e split 0.8 / 0.2
il plot che si ottiene è quello

con dati (win rate sui ritorni):
tutto il testing set (): 0.4991624790619765
primi 200 giorni: 0.605
primi 100 giorni: 0.58
primi 50 giorni: 0.66

su tutto ho 
RMSE:  76.4 (dollari ?)
MAPE:  21.8 %

su 300 giorni:
RMSE su 300 gg:  6.4 $
MAPE su 300 gg:  2.9 %

In presentazione metto prima grafico con mape e rmse su tutto facendo considerazioni, poi mi allargo con mape e rmse specifiche + winrate su meno giorni


nel firts advanced l'architettura è:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10, 10)            480       
                                                                 
 dropout (Dropout)           (None, 10, 10)            0         
                                                                 
 lstm_1 (LSTM)               (None, 10)                840       
                                                                 
 dropout_1 (Dropout)         (None, 10)                0         
                                                                 
 dense (Dense)               (None, 5)                 55        
                                                                 
 dropout_2 (Dropout)         (None, 5)                 0         
                                                                 
 dense_1 (Dense)             (None, 1)                 6         
                                                                 
=================================================================
Total params: 1,381
Trainable params: 1,381
Non-trainable params: 0
_________________________________________________________________


LTSM advanced 2: training data ridotta a 2000 giorni, arch:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10, 20)            1760      
                                                                 
 dropout (Dropout)           (None, 10, 20)            0         
                                                                 
 lstm_1 (LSTM)               (None, 20)                3280      
                                                                 
 dropout_1 (Dropout)         (None, 20)                0         
                                                                 
 dense (Dense)               (None, 5)                 105       
                                                                 
 dropout_2 (Dropout)         (None, 5)                 0         
                                                                 
 dense_1 (Dense)             (None, 1)                 6         
                                                                 
=================================================================
Total params: 5,151
Trainable params: 5,151
Non-trainable params: 0

risultati
RMSE:  10.799429328578809
MAPE:  3.1894335488381116
RMSE su 300 gg:  11.607057105021592
MAPE su 300 gg:  3.591834377775106

training di 50 epoche
ma win rate sul ritorno del giorno dopo sempre ~0.5


Il 3 ha un'architettura molto semplificata e tiene una timewindow di soli 5 gg:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 10)                480       
                                                                 
 dropout (Dropout)           (None, 10)                0         
                                                                 
 dense (Dense)               (None, 1)                 11        
                                                                 
=================================================================
Total params: 491
Trainable params: 491
Non-trainable params: 0
_________________________________________________________________
RMSE:  12.955399161548117
MAPE:  3.7480157718302904
RMSE su 300 gg:  11.019121338505466
MAPE su 300 gg:  3.3382726092879706

non si guadagna molto in winrate

semilog histo
