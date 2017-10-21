from __future__ import division
import math
import pickle


from redis import StrictRedis

redis = StrictRedis(host='localhost', port=6379, db=0)




class probabilisticEWMA:
    '''
        This is based on the paper titled probabilistic reasoning for streaming anomaly detection by Kevin M. Carter and William W. Streilein

        Redis is used for maintaining persistency between windows

        use redis to save the flag to switch between training and testing mode
    '''

    mydict = {"s1": 0, "s2": 0, "sims1": 0, "sims2": 0, "tflag": False, "count": 0, "simcount": 0}

    def __init__ ( self, term=None ):

        if term:
            #flag set to true means we are in training mode
            #set default values for alpha and beta values here
            alpha = 0.98
            beta = 0.98
            self.setParameters ( alpha, beta )
            self.setTerm ( term )

            self.setMovingParameters(  )

            #add term to list
            self.addTermToList ( term )


            self.s1 = 0
            self.s2 = 0


    def setTerm (self, term):
        '''
            add term
        '''
        self.term = term


    def removeTerm (self, term):
        '''
            remove term
        '''
        expireTime = 60 # 
        print "Remove a term"
        redis.expire (term, expireTime) 


    def addTermToList ( self, term=None):
        '''
            add terms for tracking triggers
        '''
        key = "listOfTerms"
        read_list = redis.get(key)
        mylist = []
        if read_list:
            existList = pickle.loads(read_list)
            mylist.extend ( existList )
            #save list to redis
            if term not in mylist and term is not None:
                curlist = pickle.dumps(mylist + [term])
                print term + " is saved."
                redis.set(key, curlist)
        else:
            mylistPK = pickle.dumps(mylist)
            redis.set(key, mylistPK)


    def removeTermToList ( self, term=None):
        '''
            remove terms for tracking triggers
        '''
        key = "listOfTerms"
        read_list = redis.get(key)
        mylist = []
        if read_list:
            existList = pickle.loads(read_list)
            mylist.extend ( existList )
            #save list to redis
            if term in mylist  and term is not None:
                print term + " is removed."
                mylist.remove(term)
                #remove term from redis
                self.removeTerm ( term )

                curlist = pickle.dumps(mylist)
                redis.set(key, curlist)
        else:
            mylistPK = pickle.dumps(mylist)
            redis.set(key, mylistPK)


    def getEveryTerms ( self ):
        '''
            get every terms for tracking triggers
        '''
        key = "listOfTerms"
        read_list = redis.get(key)
        mylist = []
        print "List of every terms."
        if read_list:
            existList = pickle.loads(read_list)
            mylist.extend ( existList )
        return mylist


    def deleteEveryTerms ( self ):
        '''
            delete every terms for tracking triggers
        '''
        key = "listOfTerms"
        read_list = redis.get(key)

        mylist = []
        if read_list:
            existList = pickle.loads(read_list)
            mylist.extend ( existList )
            #save list to redis
            for term in mylist:
                #remove term from redis
                if term:
                    self.removeTerm ( term )

        print "Delete every terms."
        mylistPK = pickle.dumps([])
        redis.set(key, mylistPK)


    def setParameters (self, alpha, beta):
        '''
            set parameters
        '''
        self.presetAlpha = alpha
        self.presetBeta = beta   

        
    def setData (self, data):
        self.data = data

                   
    def getData(self):
        return self.data


    def setMovingParameters( self, s1=0, s2=0, sims1=0, sims2=0, tflag=False, count=0, simcount=0 ):
        #check if term exist in the store
        read_dict = redis.get(self.term)
        if read_dict:

            mydict = {"s1": s1, "s2": s2, "sims1": sims1, "sims2": sims2, "tflag": tflag, "count": count, "simcount": simcount}

            #save dict to redis
            curdict = pickle.dumps(mydict)
            redis.set(self.term, curdict)

        else:
            p_mydict = pickle.dumps(self.mydict)
            redis.set(self.term, p_mydict)



    def getMovingParameters( self ):
        read_dict = redis.get(self.term)
        yourdict = pickle.loads(read_dict)
        return yourdict

             
    def  calcAnomalyScore(self): 
        #Training Mode
        prob_List = []

        #check if is training to exclude the code

        dictObj = self.getMovingParameters(  )

        lastCnt = dictObj["count"]

        trainFlag = dictObj["tflag"] #check if val is set
        if trainFlag:
            return
                 
        for ind in range ( 1, len( self.data ) ):
            #ind = vind + lastCnt
            if (ind == 1):
                #initialize variable
                self.s1 = self.data [ind - 1]
                self.s2 = math.pow ( self.s1, 2 )
                
            self.arc_alpha = 1 - (1.0 / (ind + 0.00001))
            alpha = 1 - (1.0 / ind )
            self.arcX_t = self.data [ind - 1]
            X_t = self.data [ind]
            Z_t = (X_t - self.arcX_t) / self.arc_alpha
                         
            P_t = (1 / math.sqrt ( 2 * math.pi )) * math.exp ( -0.5 * math.pow ( Z_t, 2 ) )

            prob_List.append (P_t) #add to list of probability score                    
            #increment the variable
                         
            self.s1 = (alpha * self.s1) + ((1 - alpha) * X_t)
            self.s2 = (alpha * self.s2) + ((1 - alpha) * math.pow ( X_t, 2 ))
                         
            self.arcX_t = self.s1
            self.arc_alpha = math.sqrt ( self.s2 - math.pow ( self.s1, 2 ) )

     
        #save the training flag to redis 
        #save  s1 to redis
        #save s2 to redis   

        count = lastCnt + len( self.data )

        self.setMovingParameters( s1=self.s1, s2=self.s2, sims1=self.s1, sims2=self.s2, tflag=True, count=count, simcount=count )  
        return prob_List
            

    def  updateAnomalyScore(self): 
        #Testing Mode
        dictObj = self.getMovingParameters(  )

        lastCnt = int (dictObj["count"])

        #get  s1 to redis self.s1
        self.s1 = float ( dictObj["s1"] )

        #get s2 to redis  self.s2
        self.s2 = float ( dictObj["s2"] )

        prob_List = []

        for ind in range ( 1, len( self.data ) ):
            #ind = vind + lastCnt
            self.arc_alpha = 1 - (1.0 / (ind + lastCnt) ) 
              
            self.arcX_t = self.data [ind - 1]
            X_t = self.data [ind]

            Z_t = (X_t - self.arcX_t) / self.arc_alpha
                         
            P_t = (1 / math.sqrt ( 2 * math.pi )) * math.exp ( - 0.5 * math.pow ( Z_t, 2 ) )
            prob_List.append ( P_t ) #add to list of probability score
                         
            alpha = self.presetAlpha
            beta = self.presetBeta
                         
            alpha = (1 - (beta * P_t)) * alpha

            self.s1 = (alpha * self.s1) + ((1 - alpha) * X_t)
            self.s2 = (alpha * self.s2) + ((1 - alpha) * math.pow ( X_t, 2 ))
                         
            self.arcX_t = self.s1
            self.arc_alpha = math.sqrt ( self.s2 - math.pow ( self.s1, 2 ) )

        #save  s1 to redis self.s1
        #save s2 to redis  self.s2

        count = lastCnt + len( self.data )

        self.setMovingParameters( s1=self.s1, s2=self.s2, sims1=self.s1, sims2=self.s2, tflag=True, count=count, simcount=count )    
                    
        return prob_List


    def predict (self, data):
        '''
            obtain index
        '''
        out = [ ]
        self.setData ( data )

        dictObj = self.getMovingParameters(  )

        statusFlag = dictObj["tflag"] 

        testing = True

        if not statusFlag: #not trained then train
            print "Training phase"
            self.calcAnomalyScore()
            testing = False

        if testing:
            print "Testing phase"
            problist = self.updateAnomalyScore( )
            out = [i for i,x in enumerate (problist) if x < 0.00000044]
        return out

