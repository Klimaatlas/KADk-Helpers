# Import sub-daily DANRA data, and bin it to daily resolution,
# applying the approach that is most appropriate for the given
# variable.
import KAPy

def import_DANRA(config,inFiles,inpID):
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
    da=KAPy.defaultImport(config, inFiles, inpID)

    #Calculate daily binned values
    if config["inputs"][inpID]["varID"]=="tas":
        da=da.resample(time='D').mean()
    elif config["inputs"][inpID]["varID"]=="tasmax":
        da=da.resample(time='D').max()
    elif config["inputs"][inpID]["varID"]=="tasmin":
        da=da.resample(time='D').min()
    elif config["inputs"][inpID]["varID"]=="pr":
        da=da.resample(time='D').sum()
        da.attrs['units']='mm/day'
    else:
        raise ValueError(f"Unknown variable ID '{config["inputs"][inpID]["varID"]}' in DANRA import.")

    return(da)
