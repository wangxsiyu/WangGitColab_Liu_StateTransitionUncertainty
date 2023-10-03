from W_Trainer.W_Training_Pipeline import W_Training_Curriculum
import sys

if __name__ == "__main__": 
    if len(sys.argv) == 4:
        pip = W_Training_Curriculum("trainer_" + sys.argv[2] + ".yaml", "curriculum_" + sys.argv[3] + ".yaml")
        pip.train(seed = int(sys.argv[1]))
    elif len(sys.argv) == 3:
        pip = W_Training_Curriculum("trainer.yaml", "curriculum_" + sys.argv[2] + ".yaml")
        pip.train(seed = int(sys.argv[1]))    
    elif len(sys.argv) == 2:
        pip = W_Training_Curriculum()
        pip.train(seed = int(sys.argv[1]))
    else:
        print('test mode on: seed = 0')
        pip = W_Training_Curriculum()
        pip.train(seed = 0)
        

    

    


