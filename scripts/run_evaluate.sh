#!/usr/bin/env bash

# cd /home/sorenh/Trained_Models2/Dipole/Model1_lr1e-3
# evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 .
# cd ..

cd /home/sorenh/Trained_Models2/Dipole/Model1_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 .
cd ..

cd /home/sorenh/Trained_Models2/Dipole/Model2_lr1e-3
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 . -i 0 -d 3
cd ..

cd /home/sorenh/Trained_Models2/Dipole/Model2_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 . -i 0 -d 3
cd ..

cd /home/sorenh/Trained_Models2/Dipole/Model3_lr1e-3
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 . -i 0 -u 64
cd ..

cd /home/sorenh/Trained_Models2/Dipole/Model3_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 . -i 0 -u 64
cd ..

cd /home/sorenh/Trained_Models2/Dipole/Model4_lr1e-3
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 . -i 0
cd ..

cd /home/sorenh/Trained_Models2/Dipole/Model4_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 0 . -i 0
cd ..



cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model1_lr1e-3
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0
cd ..

cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model1_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0
cd ..

cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model2_lr1e-3
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0 -d 3
cd ..

cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model2_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0 -d 3
cd ..

cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model3_lr1e-3
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0 -u 64
cd ..

cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model3_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0 -u 64
cd ..

cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model4_lr1e-3
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0
cd ..

cd /home/sorenh/Trained_Models2/HOMO-LUMO-Gap/Model4_lr1e-4
evaluate_model.py saved_weights/BEST.pt /Projects/Chem263Project/QM9/ 4 . -i 0
cd ..
