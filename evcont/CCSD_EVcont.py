"""
    Eigenvector continuation with CCSD trainging states     
    w. (2RDM) low-rank option 
    
    TODO 
        1. purfication  (Done)
        
    L. Wang, Sep 2025
    
"""
from evcont.ab_initio_eigenvector_continuation import approximate_multistate_lowrank_OAO
from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO, \
                                                      approximate_ground_state_OAO, \
                                                      approximate_ground_state



# for gradient and energy calculation
from evcont.ab_initio_gradients_loewdin import get_lowrank_en_with_grad_and_NAC

from evcont.electron_integral_utils import get_basis, get_integrals, get_loewdin_trafo

from evcont.low_rank_utils import reduce_2rdm, vectorize_lowrank

from pyscf import scf, gto, ao2mo, fci, lib, mcscf, mp

import numpy as np  

from ebcc import REBCC 


class CCSD_EVCont_Obj:

        
    def __init__(self,
                 
                 comp_geom_mol,   # moleclue at comp. geom 
                 nroots = 1,      # num of roots. 
                 lowrank = False,  # low-rank rep. of 2RDM 
                 comp_geom_list = None,  # for test only
                 train_geom_list = None, # for test only 
                 purify = False, # 
                 **lr_kwargs         # for lowrank compression 
                 
                ):
        
        self.comp_geom_mol = comp_geom_mol 
        self.nelec = self.comp_geom_mol.nelectron  
        self.norb = self.comp_geom_mol.nao

        self.comp_geom_list = comp_geom_list
        self.train_geom_list = train_geom_list 
        

        self.nroots = nroots
        
        if self.nroots > 1:
            raise Exception('excited states continuation not supported')
        
        self.lowrank = lowrank

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
        self.train_geom = [] 
        

        # lowrank paras 
        self.lowrank = lowrank 
        if self.lowrank is True:

            self.lr_kwargs = lr_kwargs 
            
            #iagonals of 2-cumulants ([nbra, nket, 3, norb, norb])
            self.diagonal_lr = np.zeros((1,1,3,self.norb, self.norb))

            # Low rank eigendecomposition of the rest of 2-cumulant
            # Old version: dictionary[(nbra, nket)] = (vals_trunc, vecs_trunc)
            # New version: dictionary['vals': np.array([nbra, nket, nvec]),
            #                         'vecs': np.array([nbra, nket, nvec, nao, nao])] 
            
            # key: (row_idx, col_idx)  
            # val: (what ever is needed for Hmailtonian evaluation)  
            self.vecs_lowrank = {}  

        # expansion coeff in SAO 
        self.train_mol = [] 
        
    def append_to_rdms_pro(self, mol):

        """
            new RDM and tRDM routine using Procrustes orbitals

            TODO: 
                1. low rank 
                2. 
        """

        rhf = self.get_RHF(mol)
        N_occ = mol.nelec[0]
        N_e = mol.nelectron

        num_state = len(self.train_states)  

        # CCSD in canonical MO 
        ccsd = REBCC(rhf, ansatz="CCSD")    
        ccsd.kernel()
        ccsd.solve_lambda()    
        
        # diagonal rdm in MO
        rdm1, rdm2 = ccsd.make_rdm1_f(hermitise=True), ccsd.make_rdm2_f(hermitise=True)   
        
        
        # mo-to-sao 
        U, _ = self.get_U_global(mol) 
        
        rdm1_sao = self.to_sao_1rdm(rdm1, U = U )
        rdm2_sao = self.to_sao_2rdm(rdm2, U = U )

        if num_state == 0: 
            
            self.two_rdm[0,0,:,:,:,:] = rdm2_sao.copy()  
            self.one_rdm[0,0,:,:] = rdm1_sao.copy()  
            
        else: 
            
            # diagonal RDM 
            one_rdm_old = self.one_rdm.copy()
            two_rdm_old = self.two_rdm.copy()
            ovlp_old = self.overlap.copy() 

            self.two_rdm = np.zeros([num_state+1,]*2 + [self.norb,]*4 )
            self.two_rdm[:num_state,:num_state, :,:,:,:] = two_rdm_old.copy()  
            self.overlap = np.ones([num_state+1,num_state+1]) 

            self.one_rdm = np.zeros([num_state+1,]*2 + [self.norb,]*2 )
            self.one_rdm[:num_state,:num_state, :,:] = one_rdm_old.copy()  
            self.overlap[:num_state, :num_state] = ovlp_old.copy() 

            self.one_rdm[num_state, num_state, :, : ]  = rdm1_sao.copy() 
            self.two_rdm[num_state, num_state, :, :, :, :]  = rdm2_sao.copy()  

            # tRDM in Procrustes orbtials

            for i in range(num_state):
                
                """
                    trdm2_ab(ba)_sao is internally symmetrised via: 
                    
                    symmetrize_tRDM = lambda dm: 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2,3,0,1))  

                    i.e. two swaps 
                """
                U, _ = self.get_U_global(self.train_mol[i]) 

                trdm1_ab_sao, trdm2_ab_sao = self.get_trdm_pro(self.train_states[i],
                                                               ccsd,
                                                               self.train_mol[i],
                                                               mol, 
                                                               # to SAO via A 
                                                               U
                                                               ) 
                U, _ = self.get_U_global(mol)  
                trdm1_ba_sao, trdm2_ba_sao = self.get_trdm_pro(ccsd,
                                                               self.train_states[i], 
                                                               mol, 
                                                               self.train_mol[i],
                                                               # to SAO via A 
                                                               U 
                                                               )    
                
                trdm1_ab_sym = 0.5 * (trdm1_ab_sao + trdm1_ba_sao.T) 

                # averaging Γab and Γba 
                self.one_rdm[i, num_state, :, : ] = trdm1_ab_sym 
                self.one_rdm[num_state, i, :, : ] = np.einsum('pq->qp', trdm1_ab_sym.conj()) 

                trdm2_ab_sym = 0.5 * (trdm2_ab_sao + np.einsum('pqrs->qpsr', trdm2_ba_sao))  
                
                self.two_rdm[i, num_state,:,:,:,:] = trdm2_ab_sym
                self.two_rdm[num_state, i,:,:,:,:] = np.einsum('pqrs->qpsr', trdm2_ab_sym ) 
                
                # overlap 
                self.overlap[i, num_state] = np.einsum('ii', trdm1_ab_sym) / (N_e) 
                self.overlap[num_state, i] = self.overlap[i, num_state] 
                
        self.train_states.append(ccsd.copy())
        self.train_mol.append(mol.copy())   
        self.train_en.append(ccsd.e_tot)
        

    def append_to_rdms(self, mol):
        """
            append new geometry and compute RDMs  

            Note: roots_train in FCI_EVCont is used to specify the FCI roots 
                  not relavent in CCSD

            To evaluate H in terms of low-rank tensors, one needs 
                * self.vecs_lowrank 
                * self.diagonal_lr 

            CCSD energy is required in reduce_2rdm

            On basis transformation 

            The expansion coefficient in a training geometry is given by 

            $S(\mathbf{R})^{-1/2} S(\mathbf{R_c})^{1/2} C(\mathbf{R_c}) $ being the CO expansion coeff in AO basis 
        """
        
        """
            to MO-like basis and perform CCSD
        """ 
        
        from pyscf import cc
        from numpy.linalg import eigh
        
        round = lambda x:np.round(x,7)
        rhf = self.get_RHF(mol) 

        #print('Fock\'s diagonal in canoncial basis')
        
        S = mol.intor("int1e_ovlp") 

        #print('eigenvalue of overlap matrix') 
        #print (eigh(S)[0]) 

        #print ('check orthogonality condition') 
        #print (round(rhf.mo_coeff.T @ S @ rhf.mo_coeff))
        
        F_mo =  rhf.mo_coeff.T @ rhf.get_fock() @ rhf.mo_coeff

        #print (np.diag(F_mo) ) 
        
        oao_coeff = get_basis(mol) 
        mo_like_coeff = np.einsum('ij,kj->ik',oao_coeff, self.U_global) 
        
        self.get_fock_comp(rhf,oao_coeff)  
        
        #exit() 
        
        rhf.mo_coeff = mo_like_coeff
        rhf.mo_occ = self.occ_global
        
        # needs level shift 

        ccsd = REBCC(rhf, ansatz="CCSD")    
        
        ccsd.kernel()
        ccsd.solve_lambda()     

        # pyscf in canonical basis 
        ccsd_pyscf_can = self.get_RHF(mol).CCSD()
        #ccsd_pyscf_can.level_shift = 0.25 # Ha
        ccsd_pyscf_can.kernel() 
        
        print ('energy from pyscf in canonical basis\n', ccsd_pyscf_can.e_tot) 
        print ('t1 from pyscf in canonical basis\n', self.get_t1_diag(ccsd_pyscf_can.t1))
        
        # pyscf in comp basis 

        ccsd_pyscf = cc.CCSD(rhf)
        ccsd_pyscf.level_shift = 1.64
        #ccsd_pyscf.iterative_damping = 5  
        #ccsd_pyscf.diis_start_cycle = 1
        #ccsd_pyscf.diis_space = 10

        ccsd_pyscf.kernel() 
        
        #print(ccsd_pyscf.e_tot)   
        
        rdm1_pyscf = ccsd_pyscf.make_rdm1() 
        rdm2_pyscf = ccsd_pyscf.make_rdm2()  

        ccsd_pyscf_en = self.get_ccsd_comp_en(mol,
                                              rhf,
                                              rdm1_pyscf,rdm2_pyscf,mo_like_coeff)
        
        print('energy (rdm)from pyscf in comp basis\n', ccsd_pyscf_en) 
        print('energy from pyscf in comp basis\n', ccsd_pyscf.e_tot)        
        print ('t1 from pyscf in comp basis\n', self.get_t1_diag(ccsd_pyscf.t1))
        # PYCSF RDM convention: z
        # $\Gamma_{pqrs} = \Braket{\hat{c}^\dagger_p\hat{c}^\dagger_r\hat{c}_s \hat{c}_q}$  
        rdm1, rdm2 = ccsd.make_rdm1_f(hermitise=True), ccsd.make_rdm2_f(hermitise=True)  
        
        ccsd_en_rdm = self.get_ccsd_comp_en(mol,
                                            rhf,
                                            rdm1,
                                            rdm2,
                                            mo_like_coeff)
        
        ccsd_energy = ccsd_en_rdm   
        
        print('energy from ebcc\n', ccsd_en_rdm)
        print('t1 from ebcc\n', self.get_t1_diag(ccsd.t1))

        print('\n') 
        
        num_state = len(self.train_states) 

        if self.lowrank is True: 
            
            rdm1_sao = self.to_sao_1rdm(rdm1) 
            rdm2_sao = self.to_sao_2rdm(rdm2)  
            
            if num_state == 0:
                
                self.one_rdm[0,0,:,:] = rdm1_sao.copy() 
                
                #print (self.lr_kwargs) 

                # low-rank decomp 
                lowrank_vecs, diagonals, use_joint = \
                            reduce_2rdm(rdm1_sao, 
                                        rdm2_sao, 
                                        1, 
                                        mol=mol, train_en=ccsd_energy,
                                        **self.lr_kwargs) 
                
                self.diagonal_lr[0,0,:,:,:] = diagonals.copy() 
                self.vecs_lowrank[(0,0)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint 
                
            else:
                # must update overlap first 
                self.append_1rdm_and_ovlp(num_state, rdm1, ccsd) 
                
                # copy and allocate 
                self.enlarge_lowrank_data(num_state) 

                # append diag RDM 
                lowrank_vecs, diagonals, use_joint = \
                            reduce_2rdm(rdm1_sao, 
                                        rdm2_sao, 
                                        1, 
                                        mol=mol, train_en=ccsd_energy,
                                        **self.lr_kwargs) 
                
                self.diagonal_lr[num_state,num_state,:,:,:] = diagonals.copy() 
                self.vecs_lowrank[(num_state,num_state)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint 
                
                # append tRDMs
                for i in range(num_state):
                    
                    trdm2_sym_sao = self.get_trdm_sym_sao(i, num_state, ccsd) 

                    lowrank_vecs, diagonals, use_joint = \
                            reduce_2rdm(self.one_rdm[i,num_state,:,:],        # trdm-1  
                                        trdm2_sym_sao,                        # trdm-2 
                                        self.overlap[i,num_state], 
                                        mol=mol, 
                                        train_en=ccsd_energy,                 # ? 
                                        **self.lr_kwargs) 
                    
                    self.diagonal_lr[i,num_state, :, :, :] = diagonals.copy() 
                    self.diagonal_lr[num_state,i, :, :, :] = diagonals.conj().copy() 

                    self.vecs_lowrank[(i, num_state)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint 
                    self.vecs_lowrank[(num_state, i)] = lowrank_vecs[0].conj(), lowrank_vecs[1].conj(), lowrank_vecs[2].conj(), use_joint  

        else:
            
            
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
        raise Exception('Data pruning in EC-CCSD not implemented') 
        
    """
        below are less essential/helper routines 
    """ 
    
    def get_trdm_pro(self, ccsd_A, ccsd_B, mol_A, mol_B, U ):
        """
            compute trdm via Procrustes orbitals

            returns tRDM in SAO with all in-block symmetries enforced

            A: bra 
            B: ket  (whose amplitudes will be rotated ) 
        """ 
        
        from evcont.ccsd_tRDM_utils import make_rdm1_f, make_rdm2_f 
        
        symmetrize_tRDM = lambda dm: 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2,3,0,1)) 
        
        A = self.get_D(mol_A)
        B = self.get_D(mol_B)

        N_occ = mol_A.nelec[0]

        Q_o = self.get_procrustes(A[:, :N_occ], B[:, :N_occ])
        Q_v = self.get_procrustes(A[:, N_occ:], B[:, N_occ:]) 
                

        B_t1_pro = np.einsum('ia, ji, ba -> jb',
                            ccsd_B.t1, 
                            Q_o.T,
                            Q_v.T,
                            optimize = 'optimal')
    
        B_t2_pro = np.einsum('ijab, mi, nj, ca, db -> mncd', 
                            ccsd_B.t2, 
                            Q_o.T,
                            Q_o.T, 
                            Q_v.T, 
                            Q_v.T,
                            optimize = 'optimal') 
        
        # must transform back to SAO here 
        tRDM1_AB = make_rdm1_f(
                            l1a=ccsd_A.l1,
                            l2a=ccsd_A.l2,
                            t1a=ccsd_A.t1,
                            t2a=ccsd_A.t2,
                            t1b=B_t1_pro,
                            t2b=B_t2_pro,
                         )  
    
        tRDM2_AB = make_rdm2_f(
                                l1a=ccsd_A.l1,
                                l2a=ccsd_A.l2,
                                t1a=ccsd_A.t1,
                                t2a=ccsd_A.t2,
                                t1b=B_t1_pro,
                                t2b=B_t2_pro,
                            ) 

        tRDM1_AB_sao = self.to_sao_1rdm(tRDM1_AB, U = U)
        
        tRDM2_AB_sao = self.to_sao_2rdm(tRDM2_AB, U = U )
        
        return tRDM1_AB_sao, symmetrize_tRDM(tRDM2_AB_sao)  

    def get_level_shift(self):
        """
            find the minimal level shift for CCSD 

        """    
        pass 
    
    def dbg(self, mol):

        from pyscf import cc

        rhf = self.get_RHF(mol) 
        oao_coeff = get_basis(mol)
        mo_like_coeff = np.einsum('ij,kj->ik',oao_coeff, self.U_global)  

        rhf.mo_coeff = mo_like_coeff
        rhf.mo_occ = self.occ_global

        #ccsd = REBCC(rhf, ansatz="CCSD") 
        
        #ccsd.kernel()
        #ccsd.solve_lambda()     

        # pyscf in canonical basis 
        ccsd_pyscf_can = self.get_RHF(mol).CCSD()
        #ccsd_pyscf_can.level_shift = 0.25 # Ha
        ccsd_pyscf_can.kernel() 

        print ('energy from pyscf in canonical basis\n', ccsd_pyscf_can.e_tot) 
        print ('t1 from pyscf in canonical basis\n', self.get_t1_diag(ccsd_pyscf_can.t1)) 

        ccsd_pyscf = cc.CCSD(rhf)
        #ccsd_pyscf.level_shift = 1.6

        #ccsd_pyscf.iterative_damping = 2.0
        #ccsd_pyscf.diis_start_cycle = 3
        #ccsd_pyscf.diis_space = 5

        ccsd_pyscf.kernel() 

        #print(ccsd_pyscf.e_tot)
        
        rdm1_pyscf = ccsd_pyscf.make_rdm1() 
        rdm2_pyscf = ccsd_pyscf.make_rdm2()  

        ccsd_pyscf_en = self.get_ccsd_comp_en(mol,
                                              rhf,
                                              rdm1_pyscf,rdm2_pyscf,mo_like_coeff)

        print('energy from pyscf in comp basis\n', ccsd_pyscf_en) 
        #print('energy (rdm) from pyscf in comp basis\n', ccsd_pyscf.e_tot)        
        print ('t1 from pyscf in comp basis\n', self.get_t1_diag(ccsd_pyscf.t1))        
        
    def get_fock_comp(self, rhf, sao_coeff:np.ndarray ):
        
        # sao_coeff : S^(-1/2)
        # get Fock matrix in the comp basis at new training geom 
        from numpy.linalg import eigh

        F_ao = rhf.get_fock() 

        F_sao = sao_coeff @ F_ao @ sao_coeff
        
        F_comp = self.U_global @ F_sao @ self.U_global.T 

        print('Fock matrix in comp basis') 
        print(F_comp[:5,:5])  
        print('diagonal') 
        print(np.diag(F_comp)) 
         
        # check energy levels 
        print('energy levels')
        print(eigh(F_comp)[0])
        

    def get_t1_diag(self, t1):
        # t1 diagnostics for CCSD state 
        import numpy 
        
        return numpy.sqrt(numpy.linalg.norm(t1)**2 / self.nelec) 
    
    def get_ccsd_comp_en(self, 
                         mol, 
                         rhf_comp_geom,  
                         rdm1,
                         rdm2,
                         mo_like_coeff,
                          ):
        
        """
            in MO-like basis, need to update the nuclear energy 
            to obtain the correct e_tot 
        """

        en_nuc = rhf_comp_geom.energy_nuc() 
        
        h1 = np.einsum('ai,ab,bj->ij', mo_like_coeff, rhf_comp_geom.get_hcore(), mo_like_coeff) 
        h2 = ao2mo.restore(1, ao2mo.kernel(mol, mo_like_coeff), mol.nao) 
        
        #rdm1, rdm2 = ccsd_comp_geom.make_rdm1_f(hermitise=True), ccsd_comp_geom.make_rdm2_f(hermitise=True) 

        return self.get_en_rdm(rdm1,rdm2,h1,h2,en_nuc) 
    
    def get_en_rdm(self, rdm1, rdm2, h1, h2, e_nuc ):

        en = lib.einsum("pq,qp->", h1, rdm1)
        en += lib.einsum("pqrs,pqrs->", h2, rdm2) * 0.5
        en += e_nuc

        return en  

    def get_mat_sqrt(self, mat):

        from numpy.linalg import eigh
        
        val, vec = eigh(mat)

        return  vec @ np.diag( np.sqrt(val) ) @ vec.T.conj()  

    def get_D(self, mol):

        """
            obtain D i.e. the expansion coefficients in SAO
        """ 
        rhf = scf.RHF(mol).run() 

        S = mol.intor_symmetric("int1e_ovlp")    
        
        S_sqrt = self.get_mat_sqrt(S )
        
        return S_sqrt @ rhf.mo_coeff  

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
            
            # these are not in SAO yet! 
            trdm1_ij = self.get_tRDMs_lambda('1',self.train_states[i], ccsd) 
            trdm1_ji = self.get_tRDMs_lambda('1',ccsd, self.train_states[i])
            
            # impose inter-block symmetry 
            trdm1_sym = 0.5 * (trdm1_ij.T + trdm1_ji) 
            
            # so far so good
            # assignment 
            
            self.one_rdm[num_state, i, :, :] = self.to_sao_1rdm(trdm1_sym)
            
            #self.one_rdm[i, num_state, :, :] = trdm1_sym

            self.one_rdm[i, num_state, :, :] = self.to_sao_1rdm(np.einsum('pq->qp', trdm1_sym.conj()))   
            
            # overlap 
            # np.einsum('abcc->ab',self.one_rdm)/self.nelec 
            self.overlap[i, num_state] = np.einsum('cc', self.one_rdm[i, num_state, :, :]) / self.nelec
            self.overlap[num_state, i] = self.overlap[i, num_state].conj() 
    
    def get_overlap_2rdm(self):
        
        if self.lowrank is True:
            raise Exception('low-rank continuation object does not store 2rdm explicitly')
        
        return np.einsum('abddcc->ab',self.two_rdm)/((self.nelec-1) * self.nelec)  
        
    def to_sao_1rdm(self, one_rdm, U = None):
        
        # from MO-like otbs to SAO
        if U is None:
            return np.einsum("ij,ia,jb-> ab", 
                                one_rdm, 
                                self.U_global, 
                                self.U_global, 
                                optimize="optimal") 
        else:
            return np.einsum("ij,ia,jb-> ab", 
                                one_rdm, 
                                U, U, 
                                optimize="optimal")  

    def to_sao_2rdm(self, two_rdm, U= None):
        
        # 
        if U is None: 
            return np.einsum("ijkl,ia,jb,kc,ld->abcd",
                                two_rdm,
                                self.U_global, self.U_global,
                                self.U_global, self.U_global,
                                optimize="optimal") 
        else: 
            return np.einsum("ijkl,ia,jb,kc,ld->abcd",
                                two_rdm,
                                U, U, U, U, 
                                optimize="optimal") 

    def to_mo_2rdm(self,two_rdm):
        
        return np.einsum("ijkl,ia,jb,kc,ld->abcd",
                            two_rdm,
                            self.U_global.T, self.U_global.T,
                            self.U_global.T, self.U_global.T,
                            optimize="optimal") 

    def get_RHF(self, mol): 
        
        return scf.RHF(mol).run()   

    def get_U_global(self, mol):
        """
            U = S^(1/2) C 
        """
        
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
            4. DO NOT return RDMs in SAO 
        """ 
        
        # two pairs swap symmetry 

        from evcont.ccsd_tRDM_utils import make_rdm1_f, make_rdm2_f
        
        # two swap symmetric within a single tRDM
        symmetrize_tRDM = lambda dm: 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2,3,0,1)) 
        
        if '1' in task:
            trdm1 = make_rdm1_f(
                            l1a=ccsd_i.l1,
                            l2a=ccsd_i.l2,
                            t1a=ccsd_i.t1,
                            t2a=ccsd_i.t2,
                            t1b=ccsd_j.t1,
                            t2b=ccsd_j.t2,
                        )
            #np.save('1rdm_co',self.to_sao_1rdm(trdm1)) 

        if '2' in task:    
            trdm2 = make_rdm2_f(
                            l1a=ccsd_i.l1,
                            l2a=ccsd_i.l2,
                            t1a=ccsd_i.t1,
                            t2a=ccsd_i.t2,
                            t1b=ccsd_j.t1,
                            t2b=ccsd_j.t2,
                        )
            #np.save('2rdm_co',self.to_sao_2rdm(trdm2)) 

            
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

    def test(self, 
             test_mol, 
             test_geom, 
             pivot,             # x axis, e.g. r_OH 
             data_output = 'test',  # description of simulation  
             task:list = ['ccsd','ec-ccsd'] ,
             lindep = 1e-4,     # curoff of S's eigenvalue, needs to try 
             ):
            
        """
            a built-in test routine for lightweight experiments e.g. H-chain 
            task as the 
        """
        ## just make you life easier...
        import json

        get_lambda = lambda x : np.linalg.eigh(x)[0] 
        
        print('overlap matrix')
        print(self.overlap)

        print('eigenvalues of overlap matrix')
        print(get_lambda(self.overlap))
    
        
        t1_diagnostic = lambda t1: np.sqrt(np.linalg.norm(t1)**2 / self.nelec) 
        
        x_axis = [] 
        data = {'rhf':[]} 
        
        for t in task:
            data[t] = [] 
        
        for i, mol in enumerate(test_mol):
            
            x_axis.append(test_geom[i][pivot])
            
            #rhf
            rhf = self.get_RHF(mol)
            data['rhf'].append(rhf.e_tot)
            
            # canoncial CCSD 
            if 'ccsd' in task:
                ccsd = REBCC(rhf, ansatz="CCSD")
                ccsd.kernel()
                ccsd.solve_lambda()  
                
                data['ccsd'].append(ccsd.e_tot)

                
            if 'ccsd-comp-geom' in task:
                oao_coeff = get_basis(mol)
                mo_like_coeff = np.einsum('ij,kj->ik',oao_coeff, self.U_global)
                
                h1 = np.einsum('ai,ab,bj->ij', mo_like_coeff, rhf.get_hcore(), mo_like_coeff)
                h2 = ao2mo.restore(1, ao2mo.kernel(mol, mo_like_coeff), self.norb)
                
                rhf.mo_coeff = mo_like_coeff 
                rhf.mo_occ = self.occ_global 
                
                ccsd_comp = REBCC(rhf, ansatz="CCSD")
                ccsd_comp.kernel()
                ccsd_comp.solve_lambda() 
                
                rdm1, rdm2 = ccsd_comp.make_rdm1_f(hermitise=True), ccsd_comp.make_rdm2_f(hermitise=True) 
                
                ccsd_comp_en = self.get_ccsd_comp_en(mol,
                                                     rhf,
                                                     rdm1,
                                                     rdm2,
                                                     mo_like_coeff)
                
                data['ccsd-comp-geom'].append(ccsd_comp_en) 
                
            if 'ec-ccsd' in task:
                if self.lowrank is True:
                    raise Exception('Low-rank interpolation is under construction')
                else:
                    # in SAO
                    """
                    ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

                    h1 = np.linalg.multi_dot((ao_mo_trafo.T, scf.hf.get_hcore(mol), ao_mo_trafo))
                    h2 = ao2mo.restore(1, ao2mo.kernel(mol, ao_mo_trafo), mol.nao) 
                    """
                    # lower triangle is used by default 
                    
                    en_ec_ccsd, _ = approximate_ground_state_OAO (  mol,
                                                                    self.one_rdm,
                                                                    self.two_rdm,
                                                                    self.overlap,
                                                                    hermitian=True, 
                                                                    lindep=lindep)
                
                    data['ec-ccsd'].append(en_ec_ccsd) 
        
        output = {'test_x':x_axis, 
                  'test_en':data, 
                  'train_en':self.train_en, 
                  'comp_geom_hf_en': self.comp_geom_hf_en, 
                  'comp_geom': self.comp_geom_list,
                  'train_geom': self.train_geom_list
                  } 
        
        # also wrap up 
        
        with open(data_output+'.json', 'w') as fp: 
            json.dump(output, fp)
        
    # experimental 
    def purify(self, max_ite = 50):
        """
            only purify the diagonal at this moment
            
        """
        from evcont.purification_utils import get_purified_2rdm 

        self.two_rdm = get_purified_2rdm(self.two_rdm,
                                         self.norb,
                                         self.nelec,
                                         max_ite)

    def get_procrustes(self, A, B):
    
        # A as bra, B as ket 
        
        from scipy.linalg import svd

        #M = A.T @ B 
        M = B.T @ A 
        
        #print ('overlap before', np.diag(M)) 

        U, sigma, Vh = svd(M)

        Q = U @ Vh  

        #print ('overlap after', np.diag( (B @ Q).T @ A ) )  

        return Q  
    
    