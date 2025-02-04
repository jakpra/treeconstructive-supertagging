# Tree-structured Constructive Supertagging

Models and code for ["Supertagging the Long Tail with Tree-Structured Decoding of Complex Categories"](https://arxiv.org/abs/2012.01285) by [Jakob Prange](https://prange.jakob.georgetown.domains/), [Nathan Schneider](http://people.cs.georgetown.edu/nschneid/), and [Vivek Srikumar](https://svivek.com/) (to appear in TACL)

Complete README coming soon! Please reach out with any questions (email on my [website](https://prange.jakob.georgetown.domains/)) or open an issue!

## Getting Started

Use the following code snippet to load derivations from files in CCGbank's *.auto format:

    from tree_set.util.reader import DerivationsReader
    
    dr = DerivationsReader('sample_data/wsj_0001.auto')
    derivations = dr.read_all()
    print(derivations)

This yields

    [{'ID': 'wsj_0001.1', 'PARSER': 'GOLD', 'NUMPARSE': '1', 'DERIVATION': <ccg.representation.derivation.Derivation object at 0x000001D38602C048>}, {'ID': 'wsj_0001.2', 'PARSER': 'GOLD', 'NUMPARSE': '1', 'DERIVATION': <ccg.representation.derivation.Derivation object at 0x000001D385F5E080>}]

We can inspect a single derivation object as follows:

    deriv = derivations[0]['DERIVATION']
    print(deriv.pretty_print())

which yields

    Pierre Vinken , 61    years old              , will                    join                the        board as      a          nonexecutive director Nov.                     29     .
    ______ ______ _ _____ _____ ________________ _ _______________________ ___________________ __________ _____ _______ __________ ____________ ________ ________________________ ______ _
    (N/N)  N      , (N/N) N     ((S[adj]\NP)\NP) , ((S[dcl]\NP)/(S[b]\NP)) (((S[b]\NP)/PP)/NP) (NP[nb]/N) N     (PP/NP) (NP[nb]/N) (N/N)        N        (((S\NP)\(S\NP))/N[num]) N[num] .
    ___________>A   _________>A                                                                ______________>A                    ___________________>A _____________________________>A  
    N               N                                                                          NP                                  N                     ((S\NP)\(S\NP))                  
    _____________   ___________                                            __________________________________>A         ______________________________>A                                  
    NP              NP                                                     ((S[b]\NP)/PP)                               NP                                                                
    _______________ __________________________<A                                                                ______________________________________>A                                  
    NP              (S[adj]\NP)                                                                                 PP                                                                        
                    ____________________________                           ___________________________________________________________________________>A                                  
                    (NP\NP)                                                (S[b]\NP)                                                                                                      
    __________________________________________<A                           ___________________________________________________________________________________________________________<A  
    NP                                                                     (S[b]\NP)                                                                                                      
    ______________________________________________ ___________________________________________________________________________________________________________________________________>A  
    NP                                             (S[dcl]\NP)                                                                                                                            
    __________________________________________________________________________________________________________________________________________________________________________________<A  
    S[dcl]                                                                                                                                                                                
    
Finally, we can load a whole corpus at once with the `load_derivations` function:

    import glob
    from tree_st.util.loader import load_derivations
    
    train = glob.glob('ccgbank/AUTO/0[2-9]/*.auto') 
          + glob.glob('ccgbank/AUTO/1?/*.auto') 
          + glob.glob('ccgbank/AUTO/21/*.auto')
    
    for ID, deriv in load_derivations(train):
        ...
