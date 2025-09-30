#Annual temperature range (annual delta Temp, adt)

#def adt(tasmax,tasmin):
#    annMax=tasmax.resample(time="YS").max()
#    annMin=tasmin.resample(time="YS").min()
#    out=annMax-annMin
#    out.attrs['units']="degC"
#  
#    return out
    
   
def adt(tas):
    annMax=tas.resample(time="YS").max()
    annMin=tas.resample(time="YS").min()
    out=annMax-annMin
    out.attrs['units']="degC"
  
    return out
    
