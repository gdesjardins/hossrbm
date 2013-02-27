import sys
import ast
import numpy
import os


def close_enough(x, target, margin):
    return (x <= target * (1.0 + margin) and
            x >= target * (1.0 - margin))


# Obtain the paarameters to work with
nbFilters = int(sys.argv[1])
nbFiltersMargin = int(sys.argv[2]) / 100.0
nbBlockChoices = ast.literal_eval(sys.argv[3])
nbAspectRatioChoices = ast.literal_eval(sys.argv[4])

numjobs = 0
for nbBlock in nbBlockChoices:
    
    blockSize = nbFilters / nbBlock
    
    for aspectRatio in nbAspectRatioChoices:
        
        # Compute the optimal block size with this aspect ratio
        # Solve : aspectRatio * x**2 + 0 - blockSize = 0
        optimal_x = numpy.roots([aspectRatio, 0, -blockSize])
        x = int(numpy.round(abs(optimal_x[0])))
        
        # Compute the number of filters implied by the x found
        nbFiltersApprox = nbBlock * aspectRatio * x * x
        
        # If required, adjust the number of blocks to get close to the
        # required number of filters
        if close_enough(nbFiltersApprox, nbFilters, nbFiltersMargin):
            finalNbBlock = nbBlock
        else:
            finalNbBlock = nbFilters / (aspectRatio * x * x)
            
        
        finalNbfiltersApprox = finalNbBlock * aspectRatio * x * x
        
        # If the number of filters is now close enough, launch a job with
        # those hyperparameters        
        if close_enough(finalNbfiltersApprox, nbFilters, nbFiltersMargin):
            bwg = x
            bwh = aspectRatio * bwg
            ng = bwg * finalNbBlock
            nh = bwh * finalNbBlock
            ns = finalNbfiltersApprox
            print "Scheduling numblocks=%i ar=%i:  ng=%i nh=%i ns=%i bwg=%i bwh=%i" %\
                  (ng/bwg, bwh/bwg, ng, nh, ns, bwg, bwh)
            if bwg==1 or bwh==1:
                print 'skipping'
            else:
                numjobs += 1
                os.system("schedule_hossrbm_mnistrotbackimg_experiment2_1.sh %i %i %i %i %i" %\
                          (ng, nh, ns, bwg, bwh))
        else:
            print "After adjustement, answer is still to far, it will not be considered"
            raise Exception("A valid hyperparameter combination cannot be found")
            
        print "-------------------------------------"

print 'Total number of jobs: ', numjobs
            
            
        
        
        
        
