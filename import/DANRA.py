# Import sub-daily DANRA data, and bin it to daily resolution,
# applying the approach that is most appropriate for the given
# variable.
import KAPy

def import_DANRA(inFiles,varCode,internalVarName, checks,**kwargs):
    """
    Import DANRA

    Imports DANRA data, with custom modifications made

    Parameters
    ----------
    config : _type_
        Configuration option
    inFiles : _type_
        Path to input file
    inpID : _type_
        ID of the input configuration to use
    """
    #Import using the default import functionality
    da=KAPy.defaultImport(inFiles=inFiles,
                            varCode=varCode,
                            internalVarName=internalVarName,
                            checks=checks)

    #Calculate daily binned values
    if varCode=="tas":
        da=da.resample(time='D').mean()
    elif varCode=="tasmax":
        da=da.resample(time='D').max()
    elif varCode=="tasmin":
        da=da.resample(time='D').min()
    elif varCode=="pr":
        da=da.resample(time='D').sum()
        da.attrs['units']='mm/day'
    else:
        raise ValueError(f"Unknown variable ID '{varCode}' in DANRA import.")

    return(da)
