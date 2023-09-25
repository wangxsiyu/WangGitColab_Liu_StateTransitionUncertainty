
from W_Trainer.W_Training_Pipeline import W_Training_Curriculum
import sys

if __name__ == "__main__":
    # sys.argv = [0,0]
    assert len(sys.argv) == 2
    pip = W_Training_Curriculum()
    pip.train(seed = int(sys.argv[1]))


        
    
