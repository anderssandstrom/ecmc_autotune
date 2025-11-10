require ecmccfg sandst_a "ENG_MODE=1,MASTER_ID=1"

# Master 1
${SCRIPTEXEC} ${ecmccfg_DIR}setRecordUpdateRate.cmd "RATE_MS=1"
${SCRIPTEXEC} ${ecmccfg_DIR}addSlave.cmd,      "HW_DESC=EP7211-0034_STD"
#${SCRIPTEXEC} ${ecmccfg_DIR}addSlave.cmd,      "HW_DESC=EP7211-0034_CST_STD"
#- Limit torque to 50% of motor rated torque.  Rated current = 2710mA, set to half I_MAX_MA=1355
${SCRIPTEXEC} ${ecmccfg_DIR}applyComponent.cmd "COMP=Motor-Beckhoff-AM8111-XFX0, MACROS='I_MAX_MA=1355'"
${SCRIPTEXEC} ${ecmccfg_DIR}restoreRecordUpdateRate.cmd

## X
#${SCRIPTEXEC} ${ecmccfg_DIR}addDataStorage.cmd  "DS_ID=0, SAMPLE_RATE=-1, DS_SIZE=10000"
## TRQ set
#${SCRIPTEXEC} ${ecmccfg_DIR}addDataStorage.cmd  "DS_ID=1, SAMPLE_RATE=-1, DS_SIZE=10000"
## TRQ act
#${SCRIPTEXEC} ${ecmccfg_DIR}addDataStorage.cmd  "DS_ID=2, SAMPLE_RATE=-1, DS_SIZE=10000"
## VEL act
#${SCRIPTEXEC} ${ecmccfg_DIR}addDataStorage.cmd  "DS_ID=3, SAMPLE_RATE=-1, DS_SIZE=10000"

#${SCRIPTEXEC} ${ecmccfg_DIR}loadPLCFile.cmd,    "FILE=./cfg/autotune.plc, PLC_MACROS='DRV_SID=${ECMC_EC_SLAVE_NUM}'"
#dbLoadRecords("ecmcPlcBinary.db","P=$(IOC):,PORT=MC_CPU1,ASYN_NAME=plcs.plc0.static.triggTune,REC_NAME=TrgMtn")
#dbLoadRecords("ecmcPlcAnalog.db","P=$(IOC):,PORT=MC_CPU1,ASYN_NAME=plcs.plc0.static.amplitude,REC_NAME=Amplitude")
#dbLoadRecords("ecmcPlcAnalog.db","P=$(IOC):,PORT=MC_CPU1,ASYN_NAME=plcs.plc0.static.Tc,REC_NAME=Tc")
#dbLoadRecords("ecmcPlcAnalog.db","P=$(IOC):,PORT=MC_CPU1,ASYN_NAME=plcs.plc0.static.n,REC_NAME=N")
#dbLoadRecords("ecmcPlcAnalog.db","P=$(IOC):,PORT=MC_CPU1,ASYN_NAME=plcs.plc0.static.clock,REC_NAME=Clk")




