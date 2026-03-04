"""
    Purifcation utilities
    L Wang,
    Jan 2026
"""

from purifier import purifier 
import numpy as np

def get_spinned_from_RDM(dm2):
    dm2aa_ = (dm2 - dm2.copy().transpose(0, 3, 2, 1)) / 6
    dm2bb_ = dm2aa_
    dm2ab_ = dm2 / 2 - dm2aa_
    
    return (dm2aa_, dm2ab_, dm2bb_)   

def get_purified_2rdm(rdm2, 
                      norb, 
                      nelec,
                      max_ite,):
    """
        rdm2 : 6-dimensonial array in MO 

        only apply purification to RDMs 
        (tRDM) are not considered for now 
    """
    to_D = lambda x : np.einsum('ijkl->jlik', x)  

    N = rdm2.shape[0]

    for i in range(N):
        
        print ('purification of RDM', i)

        rdm2_temp = rdm2[i,i,:,:,:,:]
        
        rdm2_aa, rdm2_ab, rdm2_bb = get_spinned_from_RDM(rdm2_temp)
        rdm1_a = np.einsum('ijkj->ik', to_D(rdm2_aa) + to_D(rdm2_ab ) )/ (nelec - 1)   

        purifier_obj = purifier(norb,
                        nelec,
                        max_ite,
                        rdm1_a,
                        rdm2_aa,
                        rdm2_ab,
                        None,
                        None,
                        None, 
                        extrapo_method = 'diis',
                        n_diis = 5, 
                        )
        
        purifier_obj.purify() 
        
        gamma_aa, gamma_ab = purifier_obj.get_final_gamma()

        # spin-traced 
        gamma_purified = gamma_aa + gamma_ab       

        gamma_purified = gamma_purified + gamma_purified.conj() 

        rdm2[i,i,:,:,:,:] = gamma_purified.copy() 

    return rdm2 
