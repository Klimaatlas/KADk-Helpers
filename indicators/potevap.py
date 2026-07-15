import numpy as np

def calculate_potential_evaporation(tas, rsds):
    """
    Calculates potential evaporation from tas and rsds using temperature-dependent
    polynomial coefficients.
    
    Parameters:
    tas (numpy.ndarray): Multi-dimensional gridcell array of daily temperatures (°C)
    rsds (numpy.ndarray): Multi-dimensional gridcell array of daily shortwave radiation (W/m^2)
    
    Returns:
    numpy.ndarray: Calculated potential evaporation array matching input dimensions
    """
    # 1. Define coefficient vectors
    cP = np.array([0.4437, 0.01429, 0.0002651, 0.000003031, 0.00000002034, 0.00000000006137])
    cM = np.array([0.5035, 0.01886, 0.0004176, 0.000005825, 0.00000004839, 0.0000000001839])
    
    # Initialize the output array with zeros
    potevap = np.zeros_like(tas, dtype=float)
    
    # 2. Identify mask indices for positive (and zero) vs negative temperatures
    ipos = tas >= 0
    ineg = tas < 0
    
    # Constant multiplier conversion factor: (24 * 3600 / 1,000,000) = 0.0864
    time_conv = (24 * 3600) / 1000000.0
    
    # 3. Process Positive Temperatures (ipos)
    if np.any(ipos):
        t_pos = tas[ipos]
        # Polynomial translation
        sP = (cP[0] + 
              2 * cP[1] * t_pos + 
              3 * cP[2] * (t_pos**2) + 
              4 * cP[3] * (t_pos**3) + 
              5 * cP[4] * (t_pos**4) + 
              6 * cP[5] * (t_pos**5))
        
        # Calculate positive potential evaporation
        potevap[ipos] = (0.7 * sP * rsds[ipos] * time_conv) / (2.465 * (sP + 0.667))
        
    # 4. Process Negative Temperatures (ineg)
    if np.any(ineg):
        t_neg = tas[ineg]
        # Polynomial translation
        sM = (cM[0] + 
              2 * cM[1] * t_neg + 
              3 * cM[2] * (t_neg**2) + 
              4 * cM[3] * (t_neg**3) + 
              5 * cM[4] * (t_neg**4) + 
              6 * cM[5] * (t_neg**5))
        
        # Calculate negative potential evaporation
        potevap[ineg] = (0.7 * sM * rsds[ineg] * time_conv) / (2.465 * (sM + 0.667))
        
    return potevap

# --- Example Usage Verification ---
if __name__ == "__main__":
    # Generating dummy climate model grid data (Time=3, Lat=2, Lon=2)
    np.random.seed(42)
    sample_tas = np.array([[[-5.0, 12.5], [0.0, -1.2]],
                           [[-2.1, 22.0], [5.3, -10.5]],
                           [[15.1, -0.5], [8.4, 19.2]]])
    
    sample_rsds = np.random.uniform(100, 250, size=sample_tas.shape)
    
    output = calculate_potential_evaporation(sample_tas, sample_rsds)
    
    print("Input Temperature Shape:", sample_tas.shape)
    print("Output Evaporation Shape:", output.shape)
    print("\nFirst timestep sample output values:\n", output[0])
