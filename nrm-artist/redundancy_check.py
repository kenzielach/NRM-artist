import numpy as np

def check_redundancy(my_design, bw):
    """ Check mask baselines for redundancy

    Checks for any redundancy in the baselines of the proposed mask design. If redundancy is above 0%, reject the mask design.

    Args:
        my_design (object): An instance of design class. The proposed mask to be tested.
        bw (float): Bandwidth of the instrument; default is to assume infinitely narrow band (optional)

    Returns:
        bool: Returns 1 if the mask has any redundancy, returns 0 if mask is fully non-redundant.
    """
    pmask = 25 # radius in pixels for 0.5m mask
    #pmask = 31 # radius in pixels for 0.65m mask
    #psc = 
    lam = 5.23*1e-6 # wavelength to use in meters; central wavelength of the band
    avg = 0
    count = 0
    fact = 2*np.pi*206265/lam
    uv_rad = my_design.hrad # need to change this
    n = 50000
    ci = 0
    for i in my_design.uv_coords:
        cj = 0
        b1 = i
        for j in my_design.uv_coords:
            b2 = j
            if ci == cj:
                cj += 1
                continue
            cj += 1
            d1 = np.sqrt((b1[0] - b2[0])**2 + (b1[1] - b2[1])**2) # both are positive
            d2 = np.sqrt((-b1[0] - b2[0])**2 + (-b1[1] - b2[1])**2) # one is negative
            d = np.min([d1, d2])
            if d < 2*uv_rad:
                test_uvs = np.random.uniform(low=0, high=2*uv_rad, size=(2,n))
                dist_b1 = np.sqrt((test_uvs[0,:] - uv_rad)**2 + (test_uvs[1,:] - uv_rad)**2)
                dist_b2 = np.sqrt((test_uvs[0,:] - (uv_rad + d))**2 + (test_uvs[1,:] - (uv_rad))**2)
                count1 = (dist_b1 <= uv_rad).sum()
                count2 = 0
                for q in range(n):
                    if dist_b1[q] <= uv_rad and dist_b2[q] < uv_rad:
                        count2 += 1
                red = 100 * np.round(count2 / count1, 2)
                avg += red
            count += 1
        ci += 1
    if count == 0:
        avg_f = 0
    else:
        avg_f = avg / count
    if avg_f > 0:
        return 1
    else:
        return 0
    
########################################################################################################
########################################################################################################
########################################################################################################