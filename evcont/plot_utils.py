"""
    a plot script for testing EC
    
    L. Wang, Oct 2025 
"""

def plot(data_path,     # .json to be read 
         plot_title,          # title of the plot            
         pivot,         # x axis , e.g. r_OH 
         train_geoms,   
         comp_geom, 
         task = ['overview','e_error'],
         y_range_energy = [-3*1e-3,3*1e-3], 
         fci_input = 'fci_full.npy', 
         suffix = '',
         ):

    import os 
    import json
    import matplotlib.pyplot as plt 

    import numpy as np 
    
    train_x = [geom[pivot] for geom in train_geoms] 

    width = 1.0
    title_fontsize = 15
    nontitle_fontsize = 12

    legend_fontsize = 10     
    fig, ax = plt.subplots(figsize=[7.2,5.4])
    

    data = None 

    with open(data_path, 'r') as fp: 
        data = json.load(fp)  

    test_range = data['test_x']
    
    hf_en = data['test_en']['rhf']
    if 'ccsd' in data['test_en']:
        ccsd_en = data['test_en']['ccsd']

    if 'ccsd-comp-geom' in data['test_en']:
        ccsd_comp_geom_en = data['test_en']['ccsd-comp-geom']
    
    ec_ccsd_en = data['test_en']['ec-ccsd']    

    if 'overview' in task:
        plt.plot(test_range,hf_en,c='gray',label='HF', linewidth = width) 
            
        """ 
            if self.compute_fci is True:
                #plt.plot(test_range,fci_en,c='k',label='FCI', linewidth = width)
                plt.plot(test_range,self.ec_fci_en,'-.',c='k',label='FCI Cont.', linewidth = width)
                #   
                np.save('fci_cont', self.ec_fci_en)
        """

        if os.path.isfile(fci_input):   
            fci_en = np.load(fci_input)
            plt.plot(test_range,fci_en,'-.',c='k',label='FCI', linewidth = width)
            
        plt.plot(test_range, ccsd_en,c='b',label='CCSD', linewidth = width)

        #if self.compute_ccsd_comp is True:
        plt.plot(test_range, ccsd_comp_geom_en,'--',c='b',label='CCSD in Comp. Geom. MO', linewidth = width)
        
        plt.plot(test_range, ec_ccsd_en,'--',c='r',label='CCSD Cont.', linewidth = width)
       
        
        plt.plot(train_x, data['train_en'],'xr',label='Training States', linewidth = width)
        #if not sao:
        
        plt.plot(comp_geom[pivot] ,data['comp_geom_hf_en'],'o',c='violet',label='Comp. Geom.', linewidth = width)
        

        plt.ylabel('Energy (Ha)', fontsize = nontitle_fontsize)

        plt.xlabel(pivot, fontsize  = nontitle_fontsize)
        #plt.xlabel(r'$r_{OH}$')
        plt.legend(fontsize = legend_fontsize )
        
        plt.title(plot_title,fontsize = title_fontsize) 
        #plt.tight_layout
            
        plt.savefig('overview-'+suffix+'.eps', dpi=500,bbox_inches='tight', format = 'eps')
    
    if 'e_error' in task:
        
        plt.clf() 
            
        plt.ylabel('Energy Diff. (Ha)', fontsize = nontitle_fontsize)
        plt.xlabel(pivot, fontsize  = nontitle_fontsize)

        # E_CCSD - E_EC
        plt.plot(test_range, 
                 np.array(ccsd_en) - np.array(ec_ccsd_en), 
                 label = 'E$_{CCSD}$ - E$_{EC}$', 
                 linewidth = width )
        
        # E_CCSD-comp - E_EC 
        plt.plot(test_range,
                 np.array(ccsd_comp_geom_en) - np.array(ec_ccsd_en),
                 label = 'E$_{CCSD-COMP}$ - E$_{EC}$', 
                 linewidth = width, 
                 
                 
                 )

        plt.ylim(y_range_energy) 
        plt.hlines(0.0, xmin = test_range[0], xmax=test_range[-1], linestyles='--', linewidth = width)

        y_min = y_range_energy[0]
        y_max = y_range_energy[1]

        for geom in train_x:
                
            plt.vlines(geom, ymin= y_min, ymax = y_max, linestyles='--', linewidth = width )

        plt.legend(fontsize = legend_fontsize )
        plt.title('Energy error, '+plot_title,fontsize = title_fontsize)     

        savefig_name =  'energy_err.eps' if suffix is None else 'energy_err-' + suffix + '.eps'
        #print (savefig_name) 
        plt.savefig(savefig_name, dpi=500,bbox_inches='tight', format = 'eps')