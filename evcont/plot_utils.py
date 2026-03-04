"""
    a plot script for testing EC
    
    L. Wang, Oct 2025 
"""

def plot(data_path_list,           # .json to be read 
         plot_title,          # title of the plot            
         pivot,         # x axis , e.g. r_OH
         x_label = 'r$_{OH}$',   
         task = ['overview','e_error'],
         y_range_energy = [-0.01,0.01], 
         fci_input = 'fci_full.npy', 
         suffix = '',
         extra_labels = ['',], 

         ):

    import os 
    import json
    import matplotlib.pyplot as plt 

    import numpy as np 
    
    #train_x = [geom[pivot] for geom in train_geoms] 

    width = 1.25
    title_fontsize = 15
    nontitle_fontsize = 12

    legend_fontsize = 10     
    fig, ax = plt.subplots(figsize=[7.2,5.4])
    
    data = None 

    read_ec_ccsd = lambda x : json.load(open(x,'r'))['test_en']['ec-ccsd']  

    with open(data_path_list[0], 'r') as fp: 
        data = json.load(fp)  
    
    comp_geom = data['comp_geom']

    # x-axis pf plot
    train_x = data['train_geom']

    test_range = data['test_x']
    
    hf_en = data['test_en']['rhf']
    if 'ccsd' in data['test_en']:
        ccsd_en = data['test_en']['ccsd']

    if 'ccsd-comp-geom' in data['test_en']:
        ccsd_comp_geom_en = data['test_en']['ccsd-comp-geom']
    
    ec_ccsd_en = data['test_en']['ec-ccsd']    

    if 'overview' in task:

        #for data_path in data_path_list: 
                
        plt.plot(test_range,hf_en,c='gray',label='HF', linewidth = width) 

        if os.path.isfile(fci_input):   
            fci_en = np.load(fci_input)
            plt.plot(test_range,fci_en,'-.',c='k',label='FCI', linewidth = width)
            
        plt.plot(test_range, ccsd_en,label='CCSD in MO', linewidth = width)

        #if self.compute_ccsd_comp is True:
        plt.plot(test_range, ccsd_comp_geom_en,'--',label='CCSD in CO', linewidth = width)
        
        # only need to loop over EC data 
        for i, data_path in enumerate(data_path_list):
            
            ec_data = None 
            
            with open(data_path, 'r') as fp: 
                ec_data = json.load(fp)
            
            ec_ccsd_en = ec_data['test_en']['ec-ccsd']     
            
            plt.plot(test_range, ec_ccsd_en,'--',label='CCSD interpolation'+' '+extra_labels[i], linewidth = width, )
        
        plt.plot(train_x, data['train_en'],'xr',label='Training geom.', linewidth = width)
        #if not sao:
        
        plt.plot(comp_geom[0] ,data['comp_geom_hf_en'],'o',c='violet',label='Comp. geom.', linewidth = width)
        
        plt.ylabel('Energy (Ha)', fontsize = nontitle_fontsize)

        plt.xlabel(x_label, fontsize  = nontitle_fontsize)
        #plt.xlabel(r'$r_{OH}$')
        plt.legend(fontsize = legend_fontsize, loc = 'upper right')
        
        plt.title(plot_title,fontsize = title_fontsize) 
        #plt.tight_layout
            
        plt.savefig('overview-'+suffix+'.jpg', dpi=500,bbox_inches='tight', format = 'jpg')
    
    if 'e_error' in task:
        
        plt.clf() 
            
        plt.ylabel('$\Delta E$ (Ha)', fontsize = nontitle_fontsize)
        plt.xlabel(x_label, fontsize  = nontitle_fontsize)
        
        # E_CCSD - E_EC
        """
        plt.plot(test_range, 
                 1000*(np.array(ccsd_en) - np.array(ec_ccsd_en)), 
                 label = 'E$_{CCSD}$ - E$_{intp}$', 
                 linewidth = width )
        
        # E_CCSD-comp - E_EC 
        plt.plot(test_range,
                 1000*(np.array(ccsd_comp_geom_en) - np.array(ec_ccsd_en)),
                 label = 'E$_{CCSD-CO}$ - E$_{intp}$', 
                 linewidth = width, 
                 )    
        """
         
        # E_FCI - E_EC 

        if os.path.isfile(fci_input):   
            fci_en = np.load(fci_input)
            #plt.plot(test_range,fci_en,'-.',c='k',label='FCI', linewidth = width) 
        
            for i, data_path in enumerate(data_path_list):
                
                ec_ccsd_en = read_ec_ccsd(data_path)
                
                plt.plot(test_range,
                        np.array(fci_en) - np.array(ec_ccsd_en),
                        label = 'Interpolation, w. purification' ,
                        linewidth = width,  
                        ) 
                
                """
                    adding the benchmark for comparison 
                """

                ec_ccsd_en_impure = read_ec_ccsd('sto-3g-6-impure.json')

                plt.plot(test_range,
                        np.array(fci_en) - np.array(ec_ccsd_en_impure),
                        label = 'Interpolation, w/o purification' ,
                        linewidth = width,  
                        )

                plt.plot(test_range,
                        np.array(fci_en) - np.array(ccsd_en),
                        label = 'CCSD-MO',
                        linewidth = width,  
                        )  
                
                plt.plot(test_range,
                        np.array(fci_en) - np.array(ccsd_comp_geom_en),
                        label = 'CCSD-CO', 
                        linewidth = width,  
                        )   
                
            
        plt.ylim(y_range_energy) 
        plt.hlines(0.0, xmin = test_range[0], xmax=test_range[-1], color = 'tab:gray',  linestyles='--', linewidth = width)

        y_min = y_range_energy[0]
        y_max = y_range_energy[1]

        for geom in train_x:
                
            plt.vlines(geom, ymin= y_min, ymax = y_max, color = 'tab:gray', linestyles='--', linewidth = width )

        plt.vlines(comp_geom, ymin= y_min, ymax = y_max, color = 'tab:gray', linestyles='--', linewidth = width*2 ) 

        plt.legend(fontsize = legend_fontsize, loc = 'upper right' )
        #plt.title('Energy error, '+plot_title,fontsize = title_fontsize)     

        savefig_name =  'energy_err.eps' if suffix is None else 'energy_err-' + suffix + '.jpg'
        #print (savefig_name) 
        plt.savefig(savefig_name, dpi=500,bbox_inches='tight', format = 'jpg')
        
    if 'others' in task:
        
        pass
        
