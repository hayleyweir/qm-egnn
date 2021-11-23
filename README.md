## Predicting QM properties with Equivarient graph nerual network

Authors: Soren Holm, Hayley Weir, Michael Miller 

### Models:
            # layers |  hidden dim |    Batch size 
    Model 1:    7           128             100
    Model 2:    3           128             100
    Model 3:    7           64              100
    Model 4:    7           128             50   

Each mode was trained with a LR of 1e-3 and 1e-4.   

The models were trains to predict  
(i)  HOMO-LUMO gap  
(ii) Dipole moment

### Set up
To set up the environment run:
`$ pip install -e .`
