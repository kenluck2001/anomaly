from __future__ import division
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../ML'))

from anomaly import probabilisticEWMA
 
if __name__ == '__main__':
    data = [2, 6, 5, 4, 7, 3, 4, 6, 3, 5, 4, 2, 6, 7, 3]
    pbObj = probabilisticEWMA("boy" )
    print pbObj.predict (data)

    data = [20, 61, 105, 4, 1, 3, 4, -6, 3, 5, 4, 2, 6, 7, 3]

    print pbObj.predict (data)



    data = [2, 6, 5, 4, 7, 3, 4, 6, 3, 5, 4, 2, 6, 7, 3]
    pbObj = probabilisticEWMA("john" )
    print pbObj.predict (data)

    data = [20, 61, 105, 4, 1, 3, 4, -6, 3, 5, 4, 2, 6, 7, 3]
    #pbObj = probabilisticEWMA( )
    print pbObj.predict (data)

    data = [2, 610, 1, 4, 1, 3, 4, -6, 3, 5, 4, 2, 6, 7, 3]
    #pbObj = probabilisticEWMA( )
    print pbObj.predict (data)

    pbObj = probabilisticEWMA( )

    pbObj.addTermToList ( "kenneth")
    pbObj.addTermToList ( "emeka")

    print pbObj.getEveryTerms()

    pbObj.deleteEveryTerms ( )

    pbObj.addTermToList ( "kenneth")
    pbObj.addTermToList ( "emeka")
    pbObj.addTermToList ( "odoh")

    print pbObj.getEveryTerms()

