import random
from rendering import render

class ImageProcessor(object):

    def __init__(self, thor_params, training_set):
        self.training_set = training_set
        self.thor_params = thor_params
        
        
    def initialize(self):
        self.train
        
    

    
class BayesianSamplerWithModel(BayesianSampler):
    def __init__(self, image, get_features, get_margins):
        self.get_features = get_features
        self.get_margins = get_margins
        self.image = image
        self.initialize_state()
        self.states = []
        self.proposals = []
    
    def initialize_state(self, state=None):
        features = self.get_features(self.image)
        self.true_features = features
        self.margins = self.get_margins(self.true_features)
        if state is None:
            self.state = np.argmax(self.margins)
                 
    def propose(self):
        reference_state = np.random.multinomial(self.margins)
        proposal = self.propose_from_reference(reference_state)
        fwdprob, bakprob = self.compute_proposal_probs(proposal)
        return proposal, fwdprob, bakprob
        
    def propose_from_reference(self, reference_state):
        diff = reference - self.state
        proposal = random.random() * diff + self.state
        return proposal
        
    def compute_proposal_probs(self, proposal):
        return 0, 0
                    
    def loop(self, N):
        for ind in xrange(N):
            proposal, fwdprob, bakprob = self.propose()
            accepted = self.accept_reject(proposal, fwdprob, bakprob)
            if accepted:
                self.state = proposal
            self.states.append(self.state)
            self.proposals.append(proposal)
             
    def accept_reject(self, proposal, fwdprob, bakprob):
        proposal_score = self.lpost(proposal)
        current_score = self.lpost(self.state)
        alpha = proposal_score - current_score - fwdprob + bakprob
        r = log(random.rand())
        if alpha > r:
            return True
        else:
            return False
        
    def lpost(self, state):
        synth_image = render(state)
        synth_features = self.model.get_features(synth_image)
        score = self.llikelihood(synth_features, self.true_features) + self.lprior(state)
        return score
        
    def llikelihood(self, f1, f2):
        llik = np.sum(f1 - f2)
        return llik
        
    def lprior(self, state):
        # make this more interesting later
        lpr = 0
        return lpr
