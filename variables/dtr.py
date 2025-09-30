#Diurnal temperature range

def dtr(tasmax,tasmin):
    dtr=(tasmax-tasmin) 
    dtr.attrs["units"] = "degC"
   
    return(dtr)
    
#Skewness
def skew(tas,tasmax,tasmin):
    skew=tas-(tasmax+tasmin)/2 
    skew.attrs["units"] = "degC"
    return(skew)


#Inversion functions
def tasmax(tas,skew,dtr):
    out=tas-skew+dtr/2
    out.attrs['units']="degC"
    return(out)

def tasmin(tas,skew,dtr):
    out=tas-skew-dtr/2
    out.attrs['units']="degC"
    return(out)
