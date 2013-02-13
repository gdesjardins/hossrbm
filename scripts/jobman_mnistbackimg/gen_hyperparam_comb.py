import sys
import ast
import numpy


def close_enough(x, target, margin):
    return (x <= target * (1.0 + margin) and
            x >= target * (1.0 - margin))


# Obtain the paarameters to work with
nbFilters = int(sys.argv[1])
nbFiltersMargin = int(sys.argv[2]) / 100.0
nbBlockChoices = ast.literal_eval(sys.argv[3])
nbAspectRatioChoices = ast.literal_eval(sys.argv[4])

for nbBlock in nbBlockChoices:
    
    blockSize = nbFilters / nbBlock
    
    for aspectRatio in nbAspectRatioChoices:
        
        # Compute the optimal block size with this aspect ratio
        # Solve : aspectRatio * x**2 + 0 - blockSize = 0
        optimal_x = numpy.roots([aspectRatio, 0, -blockSize])
        x = int(numpy.round(abs(optimal_x[0])))
        
        print "Nb blocks : ", nbBlock, ", aspect_ratio : ",aspectRatio, ", x : ",x
        
        # Compute the number of filters implied by the x found
        nbFiltersApprox = nbBlock * aspectRatio * x * x
        
        print "Nb of filters pre-adjustement : ", nbFiltersApprox
        
        # If required, adjust the number of blocks to get close to the
        # required number of filters
        if close_enough(nbFiltersApprox, nbFilters, nbFiltersMargin):
            print "Before adjustement, answer is close enough : no adjustement needed"
            finalNbBlock = nbBlock
        else:
            print "Before adjustement, answer is to far : adjustement of the number of blocks is required"
            finalNbBlock = nbFilters / (aspectRatio * x * x)
            print "Number of blocks adjusted to ", finalNbBlock
            
        
        finalNbfiltersApprox = finalNbBlock * aspectRatio * x * x
        print "Nb of filters post-adjustement : ", finalNbfiltersApprox
        
        # If the number of filters is now close enough, launch a job with
        # those hyperparameters        
        if close_enough(finalNbfiltersApprox, nbFilters, nbFiltersMargin):
            print "After adjustement, answer is close enough"
            ### TODO
        else:
            print "After adjustement, answer is still to far, it will not be considered"
            raise Exception("A valid hyperparameter combination cannot be found")
            
        print "-------------------------------------"
            
            
        
        
        
        