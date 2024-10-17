########################################################################################################
########################################################################################################
########################################################################################################
# make sure input args are valid

def check_valid_nholes(nholes):
    if nholes < 3:
        raise AttributeError(f"Please enter at least 3 holes for your mask.")
    
def check_valid_hrad(hrad):
    if hrad < 0.0:
        raise AttributeError(f'Hole size is too small.')
    if hrad > 0.77:
        raise AttributeError(f'Hole size is too big.')
    
def check_valid_scope(tn):
    if tn != 'keck':
        raise AttributeError(f'Invalid telescope.')
    
def init_checks(nholes, hrad, tn):
    check_valid_nholes(nholes)
    check_valid_hrad(hrad)
    check_valid_scope(tn)

########################################################################################################
########################################################################################################
########################################################################################################