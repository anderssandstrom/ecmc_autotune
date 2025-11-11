# Repo contains some usefull scripts for controller tuning WIP

# vel_bode_working
An example of plotting a bode diagram for the closed velocity loop of an EP7211 drive.
parameters needs to be edited inside the main.py file. 

**NOTE: motor will move!! Limits switches are not checked..**

**NOTE: ecmc axis cannot be linked to axis during test, right now.. script interfaces directlly through drive PVs.**

**Trick: `r2_min` parameter might need lowering for higher freqs otherwise segment might be discarded.**

# auto_tune_wip
WIP progress for an auto tune. Not working yet..

