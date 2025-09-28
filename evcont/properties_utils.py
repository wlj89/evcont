"""
	functionalities for property calculation 
	1. dynamic IR spectrum
	2. 


	L. Wang, Sep 2025 
"""


import numpy as np


def get_finite_diff(dipole_raw, dt ):
        
    """
        get centered diff approx. of $$\frac{d\mu}{dt} $$
        
        dt in fs  
    """   
    
    res = [0 for i in dipole_raw] 
    
    res[0] = (dipole_raw[1]-dipole_raw[0]) / dt 
    
    res[-1] = (dipole_raw[-1] - dipole_raw[-2]) / dt 

    for i in range(1,len(dipole_raw)-1): 
         
        res[i] = (dipole_raw[i+1] - dipole_raw[i-1] ) / (2 * dt)
                 
    return np.array(res)    

def get_IR_spectrum(dt:float,                   # time step in fs 
                    dipole_name:str = None,     # raw dipole moment file     
                    is_finite_diff = True,      # whether using fintie diff as input signal
                    wavenum_min = 500.0,        # cm^-1
                    wavenum_max = 4000.0,       # cm^-1
                    ):
    """
		A minimal dynamic (non Hessian-based) IR spectrum routine 
	
		Two ways (among many others) to obtain IR via Wiener-Khinchi theorem 

		$$ \mu(t) \text{ is the dipole moment time series }$$
		
		1. with finite diff dipole moment (better for low freq resolution?...)
		
		$$ I(\omega) \propto \int dt\ e^{-i \omega t } \Big(\int d\omega' 
		\abs{\mathcal{F}[\dot{\mu}]}^2 e^{i\omega' t } \Big) $$

		2. without finite diff: 
		$$ I(\omega) \propto  
			\omega^2  \abs{\mathcal{F}[\mu]}^2                              
		$$

		A great extenstion for rendering Latex comment in vscode: 
		https://marketplace.visualstudio.com/items?itemName=howcasperwhat.comment-formula 
        
    """
    from scipy.fft import fft,ifft
    
    # N*3 array 
    dipole_raw = np.load(dipole_name)

    mu_x = dipole_raw[:,0]
    mu_y = dipole_raw[:,1]
    mu_z = dipole_raw[:,2]
    
    # speed of light in cm/s 
    C_in_cm = 299792458*100 
    
    if is_finite_diff is True:
        # time derivative  
        #dDipole_dt = get_finite_diff(dipole_raw, dt)    
        dmu_dt_x = get_finite_diff(mu_x, dt)     
        dmu_dt_y = get_finite_diff(mu_y, dt)
        dmu_dt_z = get_finite_diff(mu_z, dt)

        dmu_dt = [dmu_dt_x,dmu_dt_y,dmu_dt_z] 

        # number of sampling points
        N = len(dmu_dt_x)
        print(N)

        # effective freqency range 

        N_eff = N//2
        
        # $$ \abs{\mathcal{F}[\dot{\mu}]}^2 $$ 
        
        autocorr = np.array([0+0j for i in range(N)])
        
        for item in dmu_dt: 
            # remove the # of points dependence by multiplying N 
            # scipy divides the result by N in ifft  
            # ?? 
            autocorr +=  ifft(np.abs(fft(item))) 

        #autocorr *= N 
        
        spectrum_full = np.abs(fft(autocorr))[0:N_eff]
        
        # $$ f_n = \frac{n}{T} $$       
        
        # reciprocal cm   cm-1 = Hz/c 
        recip_cm_full = np.array([n / (N * dt *1e-15 * C_in_cm) for n in range(N_eff)] )
        
        # relevant wave numbers 
        range_idx = np.where(np.logical_and(recip_cm_full<=wavenum_max, recip_cm_full>=wavenum_min))

        recip_cm = recip_cm_full[range_idx] 
        spectrum = spectrum_full[range_idx]
    
        return recip_cm, spectrum

    else:   
        raise Exception('Omega squared routine is not implemented yet') 

def plot_IR(files, dt, plot_name = 'IR'):

    import matplotlib.pylab as plt  
    fig, ax = plt.subplots(figsize=[7.2,5.4]) 
    
    for file, label in files:
        recip_cm, spectrum = get_IR_spectrum(dt,
                                             file,) 
        
        plt.plot(recip_cm, spectrum, label = label) 
    
        plt.yticks([]) 
        plt.xlabel(r'wavenumber (cm$^{-1}$)')
        plt.ylabel(r'$I(\omega)$') 
    

    plt.title('IR spectrum, EC-CCSD, cc-pVDZ')
    
    plt.legend(loc = 'upper left')
    #plt.show() 
    plt.savefig(plot_name+'.jpg', dpi=500,bbox_inches='tight', format = 'jpg')  
    
        
                
