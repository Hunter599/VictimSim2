#File dedicated to store the original code from this project


#Rescuer.py 

'''
    def cluster_victims(self):
        """ this method does a naive clustering of victims per quadrant: victims in the
            upper left quadrant compose a cluster, victims in the upper right quadrant, another one, and so on.
            
            @returns: a list of clusters where each cluster is a dictionary in the format [vic_id]: ((x,y), [<vs>])
                      such as vic_id is the victim id, (x,y) is the victim's position, and [<vs>] the list of vital signals
                      including the severity value and the corresponding label"""


        # Find the upper and lower limits for x and y
        lower_xlim = sys.maxsize    
        lower_ylim = sys.maxsize
        upper_xlim = -sys.maxsize - 1
        upper_ylim = -sys.maxsize - 1

        vic = self.victims
    
        for key, values in self.victims.items():
            x, y = values[0]
            lower_xlim = min(lower_xlim, x) 
            upper_xlim = max(upper_xlim, x)
            lower_ylim = min(lower_ylim, y)
            upper_ylim = max(upper_ylim, y)
        
        # Calculate midpoints
        mid_x = lower_xlim + (upper_xlim - lower_xlim) / 2
        mid_y = lower_ylim + (upper_ylim - lower_ylim) / 2
        print(f"{self.NAME} ({lower_xlim}, {lower_ylim}) - ({upper_xlim}, {upper_ylim})")
        print(f"{self.NAME} cluster mid_x, mid_y = {mid_x}, {mid_y}")
    
        # Divide dictionary into quadrants
        upper_left = {}
        upper_right = {}
        lower_left = {}
        lower_right = {}
        
        for key, values in self.victims.items():  # values are pairs: ((x,y), [<vital signals list>])
            x, y = values[0]
            if x <= mid_x:
                if y <= mid_y:
                    upper_left[key] = values
                else:
                    lower_left[key] = values
            else:
                if y <= mid_y:
                    upper_right[key] = values
                else:
                    lower_right[key] = values
    
        return [upper_left, upper_right, lower_left, lower_right]


        def sequencing(self):
        """ Currently, this method sort the victims by the x coordinate followed by the y coordinate
            @TODO It must be replaced by a Genetic Algorithm that finds the possibly best visiting order """

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""

        new_sequences = []

        for seq in self.sequences:   # a list of sequences, being each sequence a dictionary
            seq = dict(sorted(seq.items(), key=lambda item: item[1]))
            new_sequences.append(seq)       
            #print(f"{self.NAME} sequence of visit:\n{seq}\n")

        self.sequences = new_sequences
'''

#End-Rescuer.py