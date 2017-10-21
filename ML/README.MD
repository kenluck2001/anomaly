# Real-time anomaly detection 

## Prerequisites / Libraries
* Redis

Redis is used for maintaining persistency between windows.
Change the settings of your personal redis server to include your port number and host.

redis = StrictRedis(host='localhost', port=6379, db=0)


Further work would be putting in an XML file for easy parsing.


## Usage
See the tests folder

probabilisticEWMA class allows for tracking the anomaly on a given window of a time series. The method returns the index of the anomaly in that window.

This allow contains the ability to track multiple time series at the same time due to the Redis backend storage. Each time series is uniquely identified by a term.


In this API, predict and update 
        

## References

Carter, K. M.  & Streilein, W. W. Probabilistic Reasoning For Streaming Anomaly Detection. Retrieved 06/15, 2017, from https://ll.mit.edu/mission/cybersec/publications/publication-files/full_papers/2012_08_05_Carter_IEEESSP_FP.pdf

