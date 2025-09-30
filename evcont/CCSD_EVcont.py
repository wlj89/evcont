"""
    Eigenvector continuation with CCSD trainging states     
    w. (2RDM) low-rank option 
    
    TODO 
        1. purfication  
        
    L. Wang, Sep 2025
    
"""
from evcont.ab_initio_eigenvector_continuation import approximate_multistate_lowrank_OAO
from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO

# for gradient and energy calculation
from evcont.ab_initio_gradients_loewdin import get_lowrank_en_with_grad_and_NAC

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.low_rank_utils import reduce_2rdm, vectorize_lowrank

from pyscf import scf, gto, ao2mo, fci, lib, mcscf, mp

import numpy as np  

from ebcc import REBCC 

class CCSD_EVCont_Obj:

    
    def __init__(self,
                 
                 comp_geom_mol,   # moleclue at comp. geom 
                 nroots = 1,      # num of roots. 
                 
                 lowrank = True,  # low-rank rep. of 2RDM 
                 purtify = False, # 
                 **kwargs         # for lowrank compression 
                ):
        
        self.comp_geom_mol = comp_geom_mol 
        self.nelec = self.comp_geom_mol.nelectron  
        self.norb = self.comp_geom_mol.nao

        self.nroots = nroots

        """
            U_global & occ_global: a SAO-to-MO transformation defined in the computational geometry
                which gives rise to a MO-like basis for CCSD in any training & inference geom 
        """ 
        self.U_global, self.occ_global =  self.get_U_global(self.comp_geom_mol) 

        # data for continuation kernel
        self.overlap = np.ones((1,1))
        self.one_rdm = np.zeros((1,1,self.norb,self.norb)) 
        
        # only defined when lowrank is False  
        if self.lowrank is False:
            self.two_rdm = np.zeros((1,1,self.norb, self.norb, self.norb, self.norb))   
        
        # existing ccsd states
        self.train_states = [] 
        self.train_en = []  
        
        # lowrank paras 
        self.lowrank = lowrank 
        if self.lowrank is True:
            self.kwargs = kwargs 

            #iagonals of 2-cumulants ([nbra, nket, 3, norb, norb])
            self.diagonal_lr = np.zeros((1,1,3,self.norb, self.norb))

            # Low rank eigendecomposition of the rest of 2-cumulant
            # Old version: dictionary[(nbra, nket)] = (vals_trunc, vecs_trunc)
            # New version: dictionary['vals': np.array([nbra, nket, nvec]),
            #                         'vecs': np.array([nbra, nket, nvec, nao, nao])] 
            
            # key: (row_idx, col_idx)  
            # val: (what ever is needed for Hmailtonian evaluation)  
            self.vecs_lowrank = {}  

                       
        
    def append_to_rdms(self, mol):
        """
            append new geometry and compute RDMs  

            Note: roots_train in FCI_EVCont is used to specify the FCI roots 
                  not relavent in CCSD

            To evaluate H in terms of low-rank tensors, one needs 
                * self.vecs_lowrank 
                * self.diagonal_lr 

            CCSD energy is required in reduce_2rdm  
        """
            
        """
            to MO-like basis and perform CCSD
        """ 
        rhf = self.get_RHF(mol) 
        oao_coeff = get_basis(mol)
        mo_like_coeff = np.einsum('ij,kj->ik',oao_coeff, self.U_global) 
        
        rhf.mo_coeff = mo_like_coeff    
        rhf.mo_occ = self.occ_global   
        
        ccsd = REBCC(rhf, ansatz="CCSD") 
        ccsd.kernel()
        ccsd.solve_lambda()     
        
        ccsd_energy = ccsd.e_tot 

        # diagonal rdm in pyscf convention 
        # $\Gamma_{pqrs} = \Braket{\hat{c}^\dagger_p\hat{c}^\dagger_r\hat{c}_s \hat{c}_q}$ 
        
        # needs to rotate into SAO basis! 
        rdm1, rdm2 = ccsd.make_rdm1_f(hermitise=True), ccsd.make_rdm2_f(hermitise=True)  
        
        #self.train_states.append(ccsd)
        
        """
            need to make SAO transformation more consistent! 
        """
        
        if self.lowrank is True: 
            
            rdm1_sao = self.to_sao_1rdm(rdm1) 
            rdm2_sao = self.to_sao_2rdm(rdm2)  
            
            if num_state == 0:
                
                

                self.one_rdm[0,0,:,:] = rdm1_sao.copy() 
                
                # low-rank decomp 
                lowrank_vecs, diagonals, use_joint = \
                            reduce_2rdm(rdm1_sao, 
                                        rdm2_sao, 
                                        1, 
                                        mol=mol, train_en=ccsd_energy,
                                        **self.kwargs) 
                
                self.diagonal_lr[0,0,:,:,:] = diagonals.copy() 
                self.vecs_lowrank[(0,0)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint 
                
            else:
                # must update overlap first 
                self.append_1rdm_and_ovlp(num_state, rdm1, ccsd) 
                
                # copy and allocate 
                self.enlarge_lowrank_data(num_state) 
                
                # update diag RDM 
                lowrank_vecs, diagonals, use_joint = \
                            reduce_2rdm(rdm1_sao, 
                                        rdm2_sao, 
                                        1, 
                                        mol=mol, train_en=ccsd_energy,
                                        **self.kwargs) 
                
                self.diagonal_lr[num_state,num_state,:,:,:] = diagonals.copy() 
                self.vecs_lowrank[(num_state,num_state)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint 
                
                # update tRDM
                for i in range(num_state):
                    
                    trdm2_sym_sao = self.get_trdm_sym_sao(i, num_state, ccsd) 

                    lowrank_vecs, diagonals, use_joint = \
                            reduce_2rdm(self.one_rdm[i,num_state,:,:],        # trdm-1  
                                        trdm2_sym_sao,                        # trdm-2 
                                        self.overlap[i,num_state], 
                                        mol=mol, 
                                        train_en=ccsd_energy,                 # ? 
                                        **self.kwargs) 
                    
                    self.diagonal_lr[i,num_state, :, :, :] = diagonals.copy() 
                    self.diagonal_lr[num_state,i, :, :, :] = diagonals.conj().copy() 

                    self.vecs_lowrank[(i, num_state)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint 
                    self.vecs_lowrank[(num_state, i)] = lowrank_vecs[0].conj(), lowrank_vecs[1].conj(), lowrank_vecs[2].conj(), use_joint  
                    
                    

        else:
            num_state = len(self.train_states)
            
            if num_state == 0:  
                
                self.two_rdm[0,0,:,:,:,:] = self.to_sao_2rdm(rdm2)
                self.one_rdm[0,0,:,:] = self.to_sao_1rdm(rdm1)

            else:   
                
                self.append_1rdm_and_ovlp(num_state, rdm1, ccsd)

                two_rdm_old = self.two_rdm.copy() 

                self.two_rdm = np.zeros([num_state+1,]*2 + [self.norb,]*4 )
                self.two_rdm[:num_state,:num_state, :,:,:,:] = two_rdm_old.copy() 

                # diag 2RDM
                self.two_rdm[num_state,num_state,:,:,:,:] = self.to_sao_2rdm(rdm2)
                
                # tRDM  
                for i in range(num_state):

                    trdm2_sym_sao = self.get_trdm_sym_sao(i, num_state, ccsd)
                    
                    # trdm assignment, SAO 
                    self.two_rdm[i, num_state,:,:,:,:] = trdm2_sym_sao
                    self.two_rdm[num_state, i,:,:,:,:] = np.einsum('pqrs->qpsr', trdm2_sym_sao ) 
            
        self.train_states.append(ccsd)
        self.train_en.append(ccsd_energy)   
        
    def prune_datapoints(self, keep_ids):
        # ? 
        raise Exception('Data pruning for EC-CCSD not implemented') 
        
    """
        below are less essential/helper routines 
    """

    def get_trdm_sym_sao(self, i, num_state, ccsd):
        
        """
            get properly symmetrized tRDM via lambda amplitudes in SAO 
        """
        trdm2_ij = self.get_tRDMs_lambda('2', self.train_states[i], ccsd)
        trdm2_ji = self.get_tRDMs_lambda('2', ccsd, self.train_states[i])  

        trdm2_sym = 0.5 * (trdm2_ij + np.einsum('pqrs->qpsr', trdm2_ji)) 
        
        return self.to_sao_2rdm(trdm2_sym)
        
    def enlarge_lowrank_data(self, num_state):
        
        diag_lowrank_old = self.diagonal_lr.copy() 

        self.diagonal_lr = np.zeros([num_state+1,num_state+1,3, self.norb, self.norb])
        self.diagonal_lr[:num_state, :num_state, :,:,:] = diag_lowrank_old.copy() 

    def append_1rdm_and_ovlp(self, num_state, rdm1, ccsd):
        """
            same thing with or without low-rank 
            notice that all 1rdm are in SAO 
        """
        one_rdm_old = self.one_rdm.copy() 
        ovlp_old = self.overlap.copy() 

        self.one_rdm = np.zeros([num_state+1,]*2 + [self.norb,]*2 )
        self.one_rdm[:num_state, :num_state, :, :] = one_rdm_old.copy() 
        

        self.overlap = np.ones([num_state+1,num_state+1])
        self.overlap[:num_state, :num_state] = ovlp_old.copy() 

        self.one_rdm[num_state,num_state,:,:] = self.to_sao_1rdm(rdm1).copy() 
        
        for i in range(num_state):
            
            trdm1_ij = self.get_tRDMs_lambda('1',self.train_states[i], ccsd) 
            trdm1_ji = self.get_tRDMs_lambda('1',ccsd, self.train_states[i])
            
            # impose inter-block symmetry 
            trdm1_sym = self.to_sao_1rdm(0.5 * (trdm1_ij + trdm1_ji.T)  )
            
            # assignment 
            self.one_rdm[i, num_state, :, :] = trdm1_sym.copy() 
            self.one_rdm[num_state, i, :, :] = np.einsum('pq->qp', trdm1_sym)   

            # overlap 
            # np.einsum('abcc->ab',self.one_rdm)/self.nelec 
            self.overlap[i, num_state] = np.einsum('cc', self.one_rdm[i, num_state, :, :]) / self.nelec
            self.overlap[num_state, i] = self.overlap[i, num_state].conj() 

            
    def to_sao_1rdm(self, one_rdm):
        
        return np.einsum("ij,ia,jb-> ab", 
                            one_rdm, 
                            self.U_global, 
                            self.U_global, 
                            optimize="optimal") 

    def to_sao_2rdm(self, two_rdm):
        
        # 
        return np.einsum("ijkl,ia,jb,kc,ld->abcd",
                            two_rdm,
                            self.U_global, self.U_global,
                            self.U_global, self.U_global,
                            optimize="optimal") 

    def get_RHF(self, mol): 
        
        return scf.RHF(mol).run()   

    def get_U_global(self, mol):
        
        comp_geom_rhf = self.get_RHF(self.comp_geom_mol)

        ovlp_comp = self.comp_geom_mol.intor_symmetric("int1e_ovlp")
        
        basis_OAO_comp = get_basis(self.comp_geom_mol)

        self.comp_geom_hf_en = comp_geom_rhf.e_tot 

        return np.einsum('ji,jk,kl->il',comp_geom_rhf.mo_coeff,ovlp_comp,basis_OAO_comp), comp_geom_rhf.get_occ()
        
    def get_tRDMs_lambda(self, task:str, ccsd_i, ccsd_j):
        
        """
            1. Γba should not be derived from Γab, as a CCSD ket is not a valid N-representable wave function 
            2. that's why we need lambda amplitudes
            3. evaluate ij and ji respectively, and impose symmetries at the end 
        """ 
        
        # two pairs swap symmetry 

        from ccsd_tRDM_utils import make_rdm1_f, make_rdm2_f
        
        # two swap symmetric within a single tRDM
        symmetrize_tRDM = lambda dm: 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2,3,0,1)) 
        
        if '1' in task:
            trdm1 = self.to_sao_1rdm(
                        make_rdm1_f(
                            l1a=ccsd_i.l1,
                            l2a=ccsd_i.l2,
                            t1a=ccsd_i.t1,
                            t2a=ccsd_i.t2,
                            t1b=ccsd_j.t1,
                            t2b=ccsd_j.t2,
                        )
                    )
        if '2' in task:    
            trdm2 = self.to_sao_2rdm(
                        make_rdm2_f(
                            l1a=ccsd_i.l1,
                            l2a=ccsd_i.l2,
                            t1a=ccsd_i.t1,
                            t2a=ccsd_i.t2,
                            t1b=ccsd_j.t1,
                            t2b=ccsd_j.t2,
                        )
                    )       

        if task == '1':
            return trdm1

        elif task == '2':
            
            trdm2 = symmetrize_tRDM(trdm2) 
            return trdm2 
        elif task == '12': 
        
            trdm2 = symmetrize_tRDM(trdm2) 
            return trdm1, trdm2 
        else:
            raise Exception('task must be 1, 2, or 12')

    def test():
        
        """
            a built-in test routine for lightweight experiments e.g. H-chain 
        """
        pass 
        
        
        